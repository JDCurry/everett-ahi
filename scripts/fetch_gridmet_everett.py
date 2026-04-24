"""
Fetch GridMET weather data for Everett city bbox.

Everett bbox: lat 47.88-48.02, lon -122.29 to -122.14
GridMET resolution: ~4km — expect 4-6 grid cells covering Everett.

Fetches the same 8 weather variables used in AHI training:
  tmmx, tmmn, rmin, rmax, vs, erc, pr, vpd

Saves: data/raw/gridmet_everett_{year}.parquet (per year)
       data/processed/gridmet_everett_all.parquet (combined)
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

RAW_DIR = Path("C:/Users/JDC/Desktop/everett_ahi/data/raw")
OUT_DIR = Path("C:/Users/JDC/Desktop/everett_ahi/data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Everett city center point (centroid of neighborhoods bbox)
# Using a representative point rather than bbox for pygridmet point query
EVERETT_LAT = 47.9790
EVERETT_LON = -122.2021

# AHI weather variables
VARIABLES = ["tmmx", "tmmn", "rmin", "rmax", "vs", "erc", "pr", "vpd"]

# Date range matching existing parquet (2000-2025)
START_DATE = "2000-01-01"
END_DATE = "2025-12-31"

print("Fetching GridMET data for Everett...")
print(f"  Location: ({EVERETT_LAT}, {EVERETT_LON})")
print(f"  Variables: {VARIABLES}")
print(f"  Date range: {START_DATE} to {END_DATE}")

try:
    import pygridmet as pgm

    # Point query — gets the nearest GridMET cell
    print("\nQuerying GridMET API (this may take a few minutes)...")
    data = pgm.get_bycoords(
        coords=(EVERETT_LON, EVERETT_LAT),
        dates=(START_DATE, END_DATE),
        variables=VARIABLES,
    )

    print(f"\nReceived {len(data)} daily records")
    print(f"Columns: {list(data.columns)}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"\nSample (first 5 rows):")
    print(data.head())
    print(f"\nSummary statistics:")
    print(data.describe().round(2))

    # Save
    data.to_parquet(OUT_DIR / "gridmet_everett_all.parquet")
    print(f"\nSaved gridmet_everett_all.parquet ({len(data)} rows)")

    # Also save per-year for easier debugging
    for year in data.index.year.unique():
        year_data = data[data.index.year == year]
        year_data.to_parquet(RAW_DIR / f"gridmet_everett_{year}.parquet")
    print(f"Saved {len(data.index.year.unique())} per-year files")

except Exception as e:
    print(f"\nError: {e}")
    print("\nFallback: try fetching a single year to test...")
    try:
        import pygridmet as pgm
        test = pgm.get_bycoords(
            coords=(EVERETT_LON, EVERETT_LAT),
            dates=("2024-01-01", "2024-12-31"),
            variables=["tmmx", "pr"],
        )
        print(f"Test fetch succeeded: {len(test)} rows")
        print(test.head())
    except Exception as e2:
        print(f"Test also failed: {e2}")
        sys.exit(1)

print("\nDone.")
