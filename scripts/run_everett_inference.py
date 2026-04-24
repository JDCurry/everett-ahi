"""
Run AHI v2.5 inference for Everett using city-level GridMET data.

Approach (Option B): Use existing model trained on county data, swap in
Everett-specific weather features at inference time. The model's Snohomish
county embedding captures regional geography; we replace the weather inputs
with data from Everett's actual GridMET cell instead of the county centroid
40 miles east near Monroe.

Inputs:
  - best_model.pt (v2.5, Experiment D)
  - gridmet_everett_all.parquet (9,490 daily records, 2000-2025)
  - hazard_lm_diffusion.py (model architecture)
  - inference_core.py (calibration pipeline)

Outputs:
  - data/processed/everett_predictions.parquet (daily calibrated predictions)
"""

import sys
import json
import math
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import torch

# Add model source directory to path
HAZARD_LM_DIR = Path("C:/Users/JDC/Desktop/hazard-lm")
sys.path.insert(0, str(HAZARD_LM_DIR))

from hazard_lm_diffusion import HazardLMDiffusion, HazardDiffusionConfig, create_hazard_lm_diffusion
from inference_core import (
    STATIC_FEATURE_COLS, HAZARD_TYPES, _apply_calibration, load_temperature_scales
)

DATA_DIR = Path("C:/Users/JDC/Desktop/everett_ahi/data/processed")
OUT_DIR = DATA_DIR
MODEL_PATH = HAZARD_LM_DIR / "experiments" / "outputs" / "D_seasonal_only" / "best_model.pt"
TEMP_SCALES_PATH = HAZARD_LM_DIR / "experiments" / "outputs" / "temperature_scales_D.json"

# Everett-specific constants (from Snohomish county data + city corrections)
EVERETT_LAT = 47.979
EVERETT_LON = -122.202
SNOHOMISH_COUNTY_ID = 30  # From sorted county list mod 250
WA_STATE_ID = 0
NLCD_ID = 0  # Placeholder (matches training)

# County-level static features (reasonable for Everett as largest Snohomish city)
EVERETT_STATIC = {
    "elevation": 0.0,        # Sea level — Everett is coastal
    "forest_fraction": 0.62, # From Snohomish county parquet
    "urban_fraction": 0.6,   # From Snohomish county parquet
    "pop_density": 600.2,    # From Snohomish county parquet
    "red_flag_active": 0,    # Default — override per-day if red flag data available
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


def load_model():
    """Load AHI v2.5 (Experiment D) model."""
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    config = HazardDiffusionConfig(
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        use_diffusion_attention=True,
        adaptive_diffusion_time=False,
        base_diffusion_time=0.28,
    )
    model = HazardLMDiffusion(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    print(f"  Model loaded (epoch {checkpoint.get('epoch', '?')})")
    return model


def build_gridmet_dataframe():
    """Load Everett GridMET data and rename columns to match training feature names."""
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

    # Add date-derived features
    gm["day_of_year"] = gm["date"].dt.dayofyear.astype(float)
    gm["month"] = gm["date"].dt.month.astype(float)
    gm["year"] = gm["date"].dt.year.astype(float)

    # Add Everett-specific static features
    gm["latitude"] = EVERETT_LAT
    gm["longitude"] = EVERETT_LON
    for k, v in EVERETT_STATIC.items():
        gm[k] = v

    print(f"  {len(gm)} daily records prepared")
    return gm


def build_static_tensor(row: pd.Series, pad_dim: int = 50) -> torch.Tensor:
    """Build [1, 50] static feature tensor from a GridMET row."""
    values = []
    for col in STATIC_FEATURE_COLS:
        val = row.get(col, 0.0)
        try:
            values.append(float(val) if pd.notna(val) else 0.0)
        except (ValueError, TypeError):
            values.append(0.0)
    while len(values) < pad_dim:
        values.append(0.0)
    return torch.tensor(values[:pad_dim], dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def run_inference(model, gm: pd.DataFrame, temperatures: dict, batch_size: int = 256):
    """Run batch inference over all daily records."""
    print(f"\nRunning inference over {len(gm)} days (batch_size={batch_size})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    results = []
    total = len(gm)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_rows = gm.iloc[start:end]

        # Build static tensors
        static_list = []
        months = []
        for _, row in batch_rows.iterrows():
            static_list.append(build_static_tensor(row))
            months.append(int(row["month"]))

        static_cont = torch.cat(static_list, dim=0).to(device)
        bs = static_cont.size(0)

        # Temporal: zero-filled (matching training — no lag features in v2.5)
        temporal = torch.zeros(bs, 14, 20, dtype=torch.float32, device=device)

        # IDs: same for all Everett rows
        region_ids = torch.full((bs,), SNOHOMISH_COUNTY_ID, dtype=torch.long, device=device)
        state_ids = torch.full((bs,), WA_STATE_ID, dtype=torch.long, device=device)
        nlcd_ids = torch.full((bs,), NLCD_ID, dtype=torch.long, device=device)

        # Forward pass
        outputs = model(static_cont, temporal, region_ids, state_ids, nlcd_ids)

        # Extract and calibrate
        # Two calibration modes:
        #   "full" — standard pipeline (T-scaling + seasonal + ceiling)
        #   "city" — skip T-sharpening, apply seasonal bias + ceiling to raw logit
        # City mode prevents ceiling-saturation when running single-point inference
        for i in range(bs):
            row_date = batch_rows.iloc[i]["date"]
            month = months[i]
            row_result = {"date": row_date, "month": month}

            for h in HAZARD_TYPES:
                raw_logit = float(outputs[f"{h}_logits"][i].cpu())
                raw_prob = float(outputs[f"{h}_prob"][i].cpu())

                # Full calibration (county-level pipeline)
                calibrated_full = _apply_calibration(raw_logit, h, month, temperatures)

                # City-level calibration: no T-sharpening, just seasonal + ceiling
                city_logit = raw_logit
                # Apply seasonal bias
                from inference_core import SEASONAL_LOGIT_BIAS, _get_ceiling
                if month and 1 <= month <= 12 and h in SEASONAL_LOGIT_BIAS:
                    city_logit += SEASONAL_LOGIT_BIAS[h].get(month, 0.0)
                # Seismic base-rate bias
                if h == "seismic":
                    city_logit += -1.5
                city_prob = 1.0 / (1.0 + math.exp(-city_logit))
                city_prob = min(city_prob, _get_ceiling(h, month))
                city_prob = max(0.0, city_prob)

                row_result[f"{h}_logit"] = round(raw_logit, 4)
                row_result[f"{h}_raw"] = round(raw_prob, 4)
                row_result[f"{h}_calibrated"] = round(calibrated_full, 4)
                row_result[f"{h}_city"] = round(city_prob, 4)

            results.append(row_result)

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{total} days...")

    return pd.DataFrame(results)


def main():
    # Load model
    model = load_model()

    # Load temperature scales
    temperatures = load_temperature_scales(str(TEMP_SCALES_PATH))

    # Build feature dataframe
    gm = build_gridmet_dataframe()

    # Run inference
    predictions = run_inference(model, gm, temperatures)

    # Merge with GridMET for context
    predictions["date"] = pd.to_datetime(predictions["date"])
    gm_slim = gm[["date", "tmmx", "tmmn", "pr", "vs", "erc", "vpd"]].copy()
    merged = predictions.merge(gm_slim, on="date", how="left")

    # Save
    out_path = OUT_DIR / "everett_predictions.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path} ({len(merged)} rows)")

    # Summary
    print("\n=== Prediction Summary (City Calibration) ===")
    for h in HAZARD_TYPES:
        col = f"{h}_city"
        print(f"  {h}: mean={merged[col].mean():.4f}, "
              f"max={merged[col].max():.4f}, "
              f"p95={merged[col].quantile(0.95):.4f}, "
              f"p05={merged[col].quantile(0.05):.4f}")

    # Monthly averages
    print("\n=== Monthly Averages (City Calibration) ===")
    monthly = merged.groupby("month")[[f"{h}_city" for h in HAZARD_TYPES]].mean()
    monthly.columns = HAZARD_TYPES
    print(monthly.round(4).to_string())

    # Show a sample: April 2025 as proxy for current
    april = merged[(merged["date"].dt.month == 4) & (merged["date"].dt.year == 2025)]
    if len(april) > 0:
        print(f"\n=== Sample: April 2025 (proxy for current conditions) ===")
        for h in HAZARD_TYPES:
            print(f"  {h}: city={april[f'{h}_city'].mean():.4f}, "
                  f"raw={april[f'{h}_raw'].mean():.4f}")


if __name__ == "__main__":
    main()
