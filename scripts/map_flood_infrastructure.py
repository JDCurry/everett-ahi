"""
Spatial join: map drainage basins and flood levees to Everett neighborhoods.

Produces:
  - data/processed/neighborhood_flood_exposure.json
  - Enriches neighborhood_profiles.json with flood infrastructure data
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt

warnings.filterwarnings("ignore")

DATA_DIR = Path("C:/Users/JDC/Desktop/everett_data")
OUT_DIR = Path("C:/Users/JDC/Desktop/everett_ahi/data/processed")

# ── Load neighborhoods ──────────────────────────────────────────────────
print("Loading neighborhoods...")
neighborhoods = gpd.read_file(DATA_DIR / "Neighborhoods_20260413.geojson")
neighborhoods = neighborhoods.rename(columns={"name": "neighborhood"})
neighborhoods["neighborhood"] = neighborhoods["neighborhood"].replace(
    {"PINEHURST": "PINEHURST-BEVERLY PARK"}
)
neighborhoods = neighborhoods.to_crs("EPSG:4326")
print(f"  {len(neighborhoods)} neighborhoods")

# ── Load drainage basins ────────────────────────────────────────────────
print("Loading drainage basins...")
basins_df = pd.read_csv(DATA_DIR / "Drainage_Basins_20260413.csv", low_memory=False)
print(f"  {len(basins_df)} basins, columns: {list(basins_df.columns)}")

# Parse WKT geometry
basins_df["geometry"] = basins_df["the_geom"].apply(shapely_wkt.loads)
basins = gpd.GeoDataFrame(basins_df, geometry="geometry", crs="EPSG:4326")
basins = basins.drop(columns=["the_geom"])
print(f"  Basin names: {basins['NAME'].tolist()}")

# ── Load flood levees ──────────────────────────────────────────────────
print("\nLoading flood levees...")
levees_df = pd.read_csv(DATA_DIR / "Flood_Levee_20260413.csv", low_memory=False)
levees_df["geometry"] = levees_df["the_geom"].apply(shapely_wkt.loads)
levees = gpd.GeoDataFrame(levees_df, geometry="geometry", crs="EPSG:4326")
levees = levees.drop(columns=["the_geom"])
print(f"  {len(levees)} levee segments")
if "LEVEE_NM" in levees.columns:
    print(f"  Levee names: {levees['LEVEE_NM'].dropna().unique().tolist()}")
if "LEVEE_STAT" in levees.columns:
    print(f"  Statuses: {levees['LEVEE_STAT'].value_counts().to_dict()}")

# ── Spatial join: basins ↔ neighborhoods ────────────────────────────────
print("\n=== Spatial join: basins × neighborhoods ===")
# Use overlay intersection to find which basins overlap which neighborhoods
basin_nbr = gpd.overlay(neighborhoods[["neighborhood", "geometry"]],
                         basins[["NAME", "geometry"]].rename(columns={"NAME": "basin_name"}),
                         how="intersection")

# Calculate overlap area (project to metric CRS for area calc)
basin_nbr_proj = basin_nbr.to_crs("EPSG:32610")  # UTM Zone 10N
basin_nbr["overlap_acres"] = basin_nbr_proj.geometry.area / 4046.86  # sq meters to acres

basin_summary = basin_nbr.groupby("neighborhood").agg(
    basins=("basin_name", lambda x: sorted(x.unique().tolist())),
    num_basins=("basin_name", "nunique"),
    total_basin_overlap_acres=("overlap_acres", "sum"),
).reset_index()

print(f"\nBasin coverage by neighborhood:")
for _, row in basin_summary.iterrows():
    print(f"  {row['neighborhood']}: {row['num_basins']} basins ({row['total_basin_overlap_acres']:.0f} acres) — {row['basins']}")

# ── Spatial join: levees ↔ neighborhoods ────────────────────────────────
print("\n=== Spatial join: levees × neighborhoods ===")
# Buffer levee lines slightly for intersection with polygons
levees_buffered = levees.copy()
levees_proj = levees_buffered.to_crs("EPSG:32610")
levees_proj["geometry"] = levees_proj.geometry.buffer(50)  # 50m buffer
levees_buffered = levees_proj.to_crs("EPSG:4326")

levee_cols = ["geometry"]
if "LEVEE_NM" in levees.columns:
    levee_cols.append("LEVEE_NM")
if "LEVEE_STAT" in levees.columns:
    levee_cols.append("LEVEE_STAT")
if "WTR_NM" in levees.columns:
    levee_cols.append("WTR_NM")

levee_nbr = gpd.sjoin(neighborhoods[["neighborhood", "geometry"]],
                       levees_buffered[levee_cols],
                       how="inner", predicate="intersects")

if len(levee_nbr) > 0:
    levee_summary = levee_nbr.groupby("neighborhood").agg(
        levee_names=("LEVEE_NM", lambda x: sorted(x.dropna().unique().tolist())) if "LEVEE_NM" in levee_nbr.columns else ("index_right", "count"),
        num_levee_segments=("index_right", "count"),
    ).reset_index()
    print(f"\nLevee proximity by neighborhood:")
    for _, row in levee_summary.iterrows():
        print(f"  {row['neighborhood']}: {row['num_levee_segments']} segments — {row.get('levee_names', [])}")
else:
    levee_summary = pd.DataFrame(columns=["neighborhood", "levee_names", "num_levee_segments"])
    print("  No levee-neighborhood intersections found")

# ── Assemble flood exposure data ────────────────────────────────────────
print("\n=== Building flood exposure profiles ===")
flood_exposure = {}
for _, nbr in neighborhoods.iterrows():
    name = nbr["neighborhood"]
    basin_row = basin_summary[basin_summary["neighborhood"] == name]
    levee_row = levee_summary[levee_summary["neighborhood"] == name] if len(levee_summary) > 0 else pd.DataFrame()

    flood_exposure[name] = {
        "drainage_basins": basin_row.iloc[0]["basins"] if len(basin_row) > 0 else [],
        "num_basins": int(basin_row.iloc[0]["num_basins"]) if len(basin_row) > 0 else 0,
        "basin_overlap_acres": round(float(basin_row.iloc[0]["total_basin_overlap_acres"]), 1) if len(basin_row) > 0 else 0,
        "has_levee_protection": len(levee_row) > 0,
        "levee_segments": int(levee_row.iloc[0]["num_levee_segments"]) if len(levee_row) > 0 else 0,
    }

# Save standalone flood exposure
with open(OUT_DIR / "neighborhood_flood_exposure.json", "w") as f:
    json.dump(flood_exposure, f, indent=2)
print(f"\nSaved neighborhood_flood_exposure.json")

# ── Enrich neighborhood_profiles.json ───────────────────────────────────
profiles_path = OUT_DIR / "neighborhood_profiles.json"
with open(profiles_path) as f:
    profiles = json.load(f)

for name, exposure in flood_exposure.items():
    if name in profiles["neighborhoods"]:
        profiles["neighborhoods"][name]["flood_exposure"] = exposure

with open(profiles_path, "w") as f:
    json.dump(profiles, f, indent=2)

print(f"Enriched neighborhood_profiles.json with flood exposure data")
print("\nDone.")
