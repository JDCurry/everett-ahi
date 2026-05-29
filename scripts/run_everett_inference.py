"""
Run AHI Round 3 inference for Everett using city-level GridMET data.

Uses the deployed PNW ONNX model (Round 3, 50 features) with WA state
calibration files. Replaces the old v2.5 PyTorch inference pipeline.

Approach: Use the PNW regional model trained on county data, swap in
Everett-specific weather features at inference time. The model's Snohomish
county embedding captures regional geography; we replace the weather inputs
with data from Everett's actual GridMET cell (47.979N, 122.202W) instead
of the county centroid 40 miles east near Monroe.

Feature vector (50 dims, Round 3 layout):
  [0-20]  GridMET weather + static (from Everett GridMET cell + overrides)
  [21-24] CW3E AR slots (zero-filled, same as production deployment)
  [25-29] NFHL flood zones (from Snohomish county data)
  [30-31] SNODAS slots (zero-filled)
  [32-38] USDM drought slots (zero-filled)
  [39-43] USGS streamflow slots (zero-filled)
  [44-49] WUI (from Snohomish county data)

Calibration pipeline (per WA state files):
  1. raw_logit / T               (temperature_scales.json)
  2. + county_seasonal_bias      (county_seasonal_bias.json, Snohomish)
  3. sigmoid
  4. min(p, seasonal_ceiling)    (base_rate_ceiling.json)

Inputs:
  - PNW ONNX model (ahi/models/pnw/model.onnx)
  - WA calibration files (ahi/states/WA/*.json)
  - gridmet_everett_all.parquet (9,490 daily records, 2000-2025)

Outputs:
  - data/processed/everett_predictions.parquet (daily calibrated predictions)

Author: Joshua D. Curry
"""
import json
import math
from pathlib import Path
from datetime import date
import time

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
AHI_DEPLOY = Path(r"C:\Users\JDC\Documents\GitHub\ahi")
MODEL_PATH = AHI_DEPLOY / "models" / "pnw" / "model.onnx"
WA_STATE_DIR = AHI_DEPLOY / "states" / "WA"

DATA_DIR = Path(r"C:\Users\JDC\Desktop\ahi-platform\everett_ahi\data\processed")
OUT_DIR = DATA_DIR

HAZARD_TYPES = ['fire', 'flood', 'wind', 'winter', 'seismic']

# Round 3 feature layout (must match STATIC_FEATURE_COLS in inference_onnx.py)
STATIC_FEATURE_COLS = [
    # [0-20] GridMET + static
    'latitude', 'longitude', 'day_of_year', 'month', 'year',
    'tmmx', 'tmmn', 'rmin', 'rmax', 'vs', 'erc', 'pr', 'vpd',
    'red_flag_active', 'tmmx_3d_mean', 'pr_3d_mean', 'vs_3d_mean',
    'elevation', 'forest_fraction', 'urban_fraction', 'pop_density',
    # [21-24] CW3E AR (zero-filled at inference)
    'ar_ivt_max', 'ar_iwv_max', 'ar_active', 'ar_scale',
    # [25-29] NFHL flood zones (static)
    'nfhl_sfha_frac', 'nfhl_v_frac', 'nfhl_x_frac', 'nfhl_sfha_km2', 'nfhl_v_km2',
    # [30-31] SNODAS (zero-filled at inference)
    'snodas_swe_mean', 'snodas_depth_mean',
    # [32-38] USDM drought (zero-filled at inference)
    'usdm_intensity', 'usdm_none_frac', 'usdm_d0_frac', 'usdm_d1_frac',
    'usdm_d2_frac', 'usdm_d3_frac', 'usdm_d4_frac',
    # [39-43] USGS streamflow (zero-filled at inference)
    'usgs_log_q_mean', 'usgs_log_q_max', 'usgs_log_gh_mean', 'usgs_log_gh_max',
    'usgs_n_sites',
    # [44-49] WUI (static)
    'wui_frac', 'wui_intermix_frac', 'wui_interface_frac',
    'wui_veg_frac', 'wui_veg_cover_mean', 'wui_huden_log',
]

# ── Everett constants ──────────────────────────────────────────────────────
EVERETT_LAT = 47.979
EVERETT_LON = -122.202
SNOHOMISH_COUNTY_ID = 30   # From sorted WA county list mod 250
WA_STATE_ID = 0

# Static overrides for Everett (Snohomish county parquet has 0 for some)
EVERETT_STATIC = {
    "elevation": 0.0,            # Sea level -- Everett is coastal
    "forest_fraction": 0.62,     # Snohomish county value
    "urban_fraction": 0.6,       # Snohomish county value
    "pop_density": 600.2,        # Snohomish county value
    "red_flag_active": 0,        # Default (override per-day if available)
}

# NFHL flood zone values (Snohomish county -- all zero in deployed parquet)
SNOHOMISH_NFHL = {
    "nfhl_sfha_frac": 0.0,
    "nfhl_v_frac": 0.0,
    "nfhl_x_frac": 0.0,
    "nfhl_sfha_km2": 0.0,
    "nfhl_v_km2": 0.0,
}

# WUI values (from Snohomish county inference parquet)
SNOHOMISH_WUI = {
    "wui_frac": 0.1900,
    "wui_intermix_frac": 0.1445,
    "wui_interface_frac": 0.0455,
    "wui_veg_frac": 0.8888,
    "wui_veg_cover_mean": 79.4087,
    "wui_huden_log": 1.9223,
}

# GridMET column name mapping (pygridmet adds units to column names)
GRIDMET_COL_MAP = {
    "tmmx (K)": "tmmx",
    "tmmn (K)": "tmmn",
    "rmin (%)": "rmin",
    "rmax (%)": "rmax",
    "vs (m/s)": "vs",
    "erc (-)": "erc",
    "pr (mm)": "pr",
    "vpd (kPa)": "vpd",
}


# ── Calibration ─────────────────────────────────────────────────────────────

def load_calibration():
    """Load all WA state calibration files."""
    # Temperature scales
    with open(WA_STATE_DIR / "temperature_scales.json", encoding="utf-8-sig") as f:
        t_doc = json.load(f)
    temperatures = {h: float(t_doc["temperatures"][h]) for h in HAZARD_TYPES}

    # Seasonal bias (state-level)
    with open(WA_STATE_DIR / "seasonal_bias.json", encoding="utf-8-sig") as f:
        sb_doc = json.load(f)
    state_biases = {}
    for h, monthly in sb_doc.get("biases", {}).items():
        if h in HAZARD_TYPES:
            state_biases[h] = {int(m): float(v) for m, v in monthly.items()}

    # County-level seasonal bias (Snohomish-specific)
    county_biases = {}
    cb_path = WA_STATE_DIR / "county_seasonal_bias.json"
    if cb_path.exists():
        with open(cb_path, encoding="utf-8-sig") as f:
            cb_doc = json.load(f)
        for h in HAZARD_TYPES:
            snoh = cb_doc.get("biases", {}).get(h, {}).get("SNOHOMISH")
            if snoh:
                county_biases[h] = {int(m): float(v) for m, v in snoh.items()}

    # Base rate ceiling
    with open(WA_STATE_DIR / "base_rate_ceiling.json", encoding="utf-8-sig") as f:
        bc_doc = json.load(f)
    base_ceiling = {h: float(bc_doc["base_rate_ceiling"][h]) for h in HAZARD_TYPES}
    seasonal_ceiling = {}
    for h, monthly in bc_doc.get("seasonal_ceiling", {}).items():
        if h in HAZARD_TYPES:
            seasonal_ceiling[h] = {int(m): float(v) for m, v in monthly.items()}

    return {
        "temperatures": temperatures,
        "state_biases": state_biases,
        "county_biases": county_biases,
        "base_ceiling": base_ceiling,
        "seasonal_ceiling": seasonal_ceiling,
    }


def apply_calibration(raw_logit: float, hazard: str, month: int, cal: dict) -> float:
    """Full county-level calibration: T-scaling + county bias + ceiling."""
    T = max(cal["temperatures"].get(hazard, 1.0), 0.01)
    scaled = raw_logit / T

    # Prefer county-level bias (Snohomish), fall back to state-level
    if hazard in cal["county_biases"] and month in cal["county_biases"][hazard]:
        bias = cal["county_biases"][hazard][month]
    elif hazard in cal["state_biases"] and month in cal["state_biases"][hazard]:
        bias = cal["state_biases"][hazard][month]
    else:
        bias = 0.0
    scaled += bias

    prob = 1.0 / (1.0 + math.exp(-scaled))

    # Seasonal ceiling (month-specific), else base ceiling
    sc = cal["seasonal_ceiling"]
    if month and 1 <= month <= 12 and hazard in sc and month in sc[hazard]:
        ceiling = sc[hazard][month]
    else:
        ceiling = cal["base_ceiling"].get(hazard, 1.0)

    return max(0.0, min(prob, ceiling))


def apply_city_calibration(raw_logit: float, hazard: str, month: int, cal: dict) -> float:
    """City-level calibration: skip T-sharpening, use county bias + ceiling.

    For single-point city inference, heavy T-sharpening can saturate probabilities
    at the ceiling. City mode applies lighter calibration: county-level bias
    (for local seasonal accuracy) + ceiling only. No temperature scaling.
    """
    scaled = raw_logit

    # Prefer county-level bias (Snohomish), fall back to state-level
    if hazard in cal["county_biases"] and month in cal["county_biases"][hazard]:
        scaled += cal["county_biases"][hazard][month]
    elif hazard in cal["state_biases"] and month in cal["state_biases"][hazard]:
        scaled += cal["state_biases"][hazard][month]

    # Seismic base-rate adjustment
    if hazard == "seismic":
        scaled += -1.5

    prob = 1.0 / (1.0 + math.exp(-scaled))

    # Same ceiling logic
    sc = cal["seasonal_ceiling"]
    if month and 1 <= month <= 12 and hazard in sc and month in sc[hazard]:
        ceiling = sc[hazard][month]
    else:
        ceiling = cal["base_ceiling"].get(hazard, 1.0)

    return max(0.0, min(prob, ceiling))


# ── Feature construction ────────────────────────────────────────────────────

def build_gridmet_dataframe():
    """Load Everett GridMET data and prepare full feature DataFrame."""
    print("Loading Everett GridMET data...")
    gm = pd.read_parquet(DATA_DIR / "gridmet_everett_all.parquet")
    gm = gm.rename(columns=GRIDMET_COL_MAP)
    gm.index.name = "date"
    gm = gm.reset_index()
    gm["date"] = pd.to_datetime(gm["date"])

    # Compute 3-day rolling means (matching training features)
    gm["tmmx_3d_mean"] = gm["tmmx"].rolling(3, min_periods=1).mean()
    gm["pr_3d_mean"] = gm["pr"].rolling(3, min_periods=1).mean()
    gm["vs_3d_mean"] = gm["vs"].rolling(3, min_periods=1).mean()

    # Date-derived features
    gm["day_of_year"] = gm["date"].dt.dayofyear.astype(float)
    gm["month"] = gm["date"].dt.month.astype(float)
    gm["year"] = gm["date"].dt.year.astype(float)

    # Everett-specific overrides
    gm["latitude"] = EVERETT_LAT
    gm["longitude"] = EVERETT_LON
    for k, v in EVERETT_STATIC.items():
        gm[k] = v

    # NFHL flood zone features (static, from Snohomish county)
    for k, v in SNOHOMISH_NFHL.items():
        gm[k] = v

    # WUI features (static, from Snohomish county)
    for k, v in SNOHOMISH_WUI.items():
        gm[k] = v

    # Zero-fill features not available at inference
    # CW3E AR [21-24], SNODAS [30-31], USDM [32-38], USGS [39-43]
    zero_fill_cols = [
        'ar_ivt_max', 'ar_iwv_max', 'ar_active', 'ar_scale',
        'snodas_swe_mean', 'snodas_depth_mean',
        'usdm_intensity', 'usdm_none_frac', 'usdm_d0_frac', 'usdm_d1_frac',
        'usdm_d2_frac', 'usdm_d3_frac', 'usdm_d4_frac',
        'usgs_log_q_mean', 'usgs_log_q_max', 'usgs_log_gh_mean',
        'usgs_log_gh_max', 'usgs_n_sites',
    ]
    for c in zero_fill_cols:
        gm[c] = 0.0

    print(f"  {len(gm)} daily records prepared")
    print(f"  Date range: {gm['date'].min()} to {gm['date'].max()}")
    return gm


def build_static_batch(df: pd.DataFrame) -> np.ndarray:
    """Build [N, 50] static feature array from DataFrame rows.

    Vectorized construction - much faster than row-by-row.
    """
    arrays = []
    for col in STATIC_FEATURE_COLS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values
        else:
            vals = np.zeros(len(df), dtype=np.float32)
        arrays.append(vals)
    return np.stack(arrays, axis=1).astype(np.float32)


# ── ONNX inference ──────────────────────────────────────────────────────────

def load_onnx_session():
    """Load PNW ONNX model."""
    import onnxruntime as ort

    print(f"Loading PNW ONNX model from {MODEL_PATH}...")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(MODEL_PATH), sess_options=opts, providers=['CPUExecutionProvider']
    )
    size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"  Loaded ({size_mb:.1f} MB)")

    # Verify inputs
    input_names = {i.name for i in session.get_inputs()}
    print(f"  Inputs: {input_names}")
    return session


def run_inference(session, gm: pd.DataFrame, cal: dict):
    """Run ONNX inference over all daily records (batch_size=1).

    The ONNX model's spatial attention reshapes are traced with batch=1,
    so we process one sample at a time. For 9,490 records this is still
    fast (~30s on CPU).
    """
    print(f"\nRunning inference over {len(gm)} days...")
    t0 = time.time()

    # Check if model expects climate_region_ids
    input_names = {i.name for i in session.get_inputs()}
    needs_crid = 'climate_region_ids' in input_names

    # Pre-build the full static array for vectorized column extraction
    static_all = build_static_batch(gm)
    months_all = gm["month"].astype(int).values
    dates_all = gm["date"].values

    # Constant tensors (reused every iteration)
    temporal = np.zeros((1, 14, 20), dtype=np.float32)
    region_ids = np.array([SNOHOMISH_COUNTY_ID], dtype=np.int64)
    state_ids = np.array([WA_STATE_ID], dtype=np.int64)
    nlcd_ids = np.array([0], dtype=np.int64)
    crid = np.array([6], dtype=np.int64)  # PNW region

    results = []
    total = len(gm)

    for i in range(total):
        static_cont = static_all[i:i+1]  # [1, 50]

        feeds = {
            'static_cont': static_cont,
            'temporal': temporal,
            'region_ids': region_ids,
            'state_ids': state_ids,
            'nlcd_ids': nlcd_ids,
        }
        if needs_crid:
            feeds['climate_region_ids'] = crid

        outputs = session.run(None, feeds)

        month = int(months_all[i])
        row_result = {"date": dates_all[i], "month": month}

        for h_idx, h in enumerate(HAZARD_TYPES):
            raw_logit = float(outputs[h_idx].flatten()[0])
            raw_prob = 1.0 / (1.0 + math.exp(-raw_logit))

            # Full calibration (county-level, production pipeline)
            calibrated = apply_calibration(raw_logit, h, month, cal)

            # City calibration (lighter, no T-sharpening)
            city = apply_city_calibration(raw_logit, h, month, cal)

            row_result[f"{h}_logit"] = round(raw_logit, 4)
            row_result[f"{h}_raw"] = round(raw_prob, 4)
            row_result[f"{h}_calibrated"] = round(calibrated, 4)
            row_result[f"{h}_city"] = round(city, 4)

        results.append(row_result)

        if (i + 1) % 2000 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:,}/{total:,}] {rate:.0f} rows/s")

    elapsed = time.time() - t0
    print(f"  Done: {total:,} rows in {elapsed:.1f}s ({total/elapsed:.0f} rows/s)")
    return pd.DataFrame(results)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Load calibration
    print("Loading WA state calibration...")
    cal = load_calibration()
    print(f"  Temperatures: {cal['temperatures']}")
    print(f"  County biases: {len(cal['county_biases'])} hazards with Snohomish data")

    # Load ONNX model
    session = load_onnx_session()

    # Build feature dataframe
    gm = build_gridmet_dataframe()

    # Run inference
    predictions = run_inference(session, gm, cal)

    # Compute risk_index columns (0-100 scale, used by dashboard)
    for h in HAZARD_TYPES:
        predictions[f"{h}_risk_index"] = (predictions[f"{h}_city"] * 100).clip(0, 100).round(1)

    # Merge with GridMET weather for dashboard context
    predictions["date"] = pd.to_datetime(predictions["date"])
    gm_slim = gm[["date", "tmmx", "tmmn", "pr", "vs", "erc", "vpd"]].copy()
    merged = predictions.merge(gm_slim, on="date", how="left")

    # Save
    out_path = OUT_DIR / "everett_predictions.parquet"
    merged.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    elapsed = time.time() - t_start
    print(f"\nSaved {out_path}")
    print(f"  {len(merged):,} rows, {len(merged.columns)} columns, {size_mb:.1f} MB")
    print(f"  Total time: {elapsed:.1f}s")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Prediction Summary (City Calibration)")
    print(f"{'='*60}")
    for h in HAZARD_TYPES:
        col = f"{h}_city"
        print(f"  {h:8s}: mean={merged[col].mean():.4f}  "
              f"max={merged[col].max():.4f}  "
              f"p95={merged[col].quantile(0.95):.4f}  "
              f"p05={merged[col].quantile(0.05):.4f}")

    print(f"\n{'='*60}")
    print("Prediction Summary (Full Calibration)")
    print(f"{'='*60}")
    for h in HAZARD_TYPES:
        col = f"{h}_calibrated"
        print(f"  {h:8s}: mean={merged[col].mean():.4f}  "
              f"max={merged[col].max():.4f}  "
              f"p95={merged[col].quantile(0.95):.4f}  "
              f"p05={merged[col].quantile(0.05):.4f}")

    # Monthly averages
    print(f"\n{'='*60}")
    print("Monthly Averages (City Calibration)")
    print(f"{'='*60}")
    monthly = merged.groupby("month")[[f"{h}_city" for h in HAZARD_TYPES]].mean()
    monthly.columns = HAZARD_TYPES
    print(monthly.round(4).to_string())

    # Sample: most recent data
    recent = merged[merged["date"] >= "2025-01-01"]
    if len(recent) > 0:
        print(f"\n{'='*60}")
        print(f"2025 YTD Averages (City Calibration, {len(recent)} days)")
        print(f"{'='*60}")
        for h in HAZARD_TYPES:
            print(f"  {h:8s}: mean={recent[f'{h}_city'].mean():.4f}  "
                  f"max={recent[f'{h}_city'].max():.4f}")


if __name__ == "__main__":
    main()
