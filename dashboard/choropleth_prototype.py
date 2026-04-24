"""
Everett AHI Neighborhood Choropleth Prototype

Renders 19 neighborhoods colored by hazard exposure metrics:
  - Total hazard-related 911 calls
  - Flood basin exposure
  - CPRI-weighted composite risk

Uses Plotly for interactive hover with neighborhood details.
"""

import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import pandas as pd

DATA_DIR = Path("C:/Users/JDC/Desktop/everett_ahi/data/processed")
EVERETT_DATA = Path("C:/Users/JDC/Desktop/everett_data")

# ── Load data ───────────────────────────────────────────────────────────
with open(DATA_DIR / "neighborhood_profiles.json") as f:
    profiles = json.load(f)

neighborhoods = gpd.read_file(EVERETT_DATA / "Neighborhoods_20260413.geojson")
neighborhoods = neighborhoods.rename(columns={"name": "neighborhood"})
neighborhoods["neighborhood"] = neighborhoods["neighborhood"].replace(
    {"PINEHURST": "PINEHURST-BEVERLY PARK"}
)

# Build dataframe from profiles
rows = []
for name, p in profiles["neighborhoods"].items():
    flood_exp = p.get("flood_exposure", {})
    rows.append({
        "neighborhood": name,
        "total_hazard_calls": p["total_hazard_calls"],
        "fire_calls": p["hazard_call_counts"].get("fire", 0),
        "flood_calls": p["hazard_call_counts"].get("flood", 0),
        "winter_calls": p["hazard_call_counts"].get("winter", 0),
        "area_acres": p["area_acres"],
        "num_basins": flood_exp.get("num_basins", 0),
        "basin_overlap_acres": flood_exp.get("basin_overlap_acres", 0),
        "has_levee": flood_exp.get("has_levee_protection", False),
        "drainage_basins": ", ".join(flood_exp.get("drainage_basins", [])),
    })

df = pd.DataFrame(rows)

# Normalize call density (calls per 100 acres)
df["call_density"] = (df["total_hazard_calls"] / df["area_acres"].replace(0, 1)) * 100
df["call_density"] = df["call_density"].round(1)

# Merge with geodataframe
gdf = neighborhoods.merge(df, on="neighborhood", how="left")

# ── Choropleth 1: Total Hazard Calls ───────────────────────────────────
fig1 = px.choropleth_mapbox(
    gdf,
    geojson=gdf.geometry.__geo_interface__,
    locations=gdf.index,
    color="total_hazard_calls",
    color_continuous_scale="YlOrRd",
    hover_name="neighborhood",
    hover_data={
        "total_hazard_calls": True,
        "fire_calls": True,
        "winter_calls": True,
        "call_density": ":.1f",
        "area_acres": ":.0f",
        "num_basins": True,
    },
    mapbox_style="carto-positron",
    center={"lat": 47.9790, "lon": -122.2021},
    zoom=11.5,
    opacity=0.7,
    title="Everett Neighborhoods — Hazard-Related 911 Calls",
)
fig1.update_layout(
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    height=700,
    coloraxis_colorbar_title="Total Calls",
)

# ── Choropleth 2: Call Density (per 100 acres) ────────────────────────
fig2 = px.choropleth_mapbox(
    gdf,
    geojson=gdf.geometry.__geo_interface__,
    locations=gdf.index,
    color="call_density",
    color_continuous_scale="Viridis",
    hover_name="neighborhood",
    hover_data={
        "call_density": ":.1f",
        "total_hazard_calls": True,
        "area_acres": ":.0f",
        "drainage_basins": True,
        "has_levee": True,
    },
    mapbox_style="carto-positron",
    center={"lat": 47.9790, "lon": -122.2021},
    zoom=11.5,
    opacity=0.7,
    title="Everett Neighborhoods — Hazard Call Density (per 100 acres)",
)
fig2.update_layout(
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    height=700,
    coloraxis_colorbar_title="Calls/100ac",
)

# ── Choropleth 3: Flood Basin Exposure ─────────────────────────────────
fig3 = px.choropleth_mapbox(
    gdf,
    geojson=gdf.geometry.__geo_interface__,
    locations=gdf.index,
    color="num_basins",
    color_continuous_scale="Blues",
    hover_name="neighborhood",
    hover_data={
        "num_basins": True,
        "basin_overlap_acres": ":.0f",
        "drainage_basins": True,
        "has_levee": True,
        "flood_calls": True,
    },
    mapbox_style="carto-positron",
    center={"lat": 47.9790, "lon": -122.2021},
    zoom=11.5,
    opacity=0.7,
    title="Everett Neighborhoods — Drainage Basin Exposure",
)
fig3.update_layout(
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    height=700,
    coloraxis_colorbar_title="# Basins",
)

# ── Save as HTML ───────────────────────────────────────────────────────
out_dir = Path("C:/Users/JDC/Desktop/everett_ahi/dashboard")

fig1.write_html(out_dir / "choropleth_hazard_calls.html")
fig2.write_html(out_dir / "choropleth_call_density.html")
fig3.write_html(out_dir / "choropleth_flood_basins.html")

print("Saved 3 choropleth maps:")
print(f"  1. {out_dir / 'choropleth_hazard_calls.html'}")
print(f"  2. {out_dir / 'choropleth_call_density.html'}")
print(f"  3. {out_dir / 'choropleth_flood_basins.html'}")

# ── Summary table ──────────────────────────────────────────────────────
print("\n=== Neighborhood Summary ===")
print(df.sort_values("call_density", ascending=False)[
    ["neighborhood", "total_hazard_calls", "call_density", "area_acres", "num_basins", "has_levee"]
].to_string(index=False))

# ── City-wide stats ────────────────────────────────────────────────────
print(f"\n=== City-Wide ===")
print(f"Total hazard calls: {df['total_hazard_calls'].sum()}")
print(f"Total buildings: {profiles['total_buildings']}")
print(f"Shelters: {len(profiles['shelters'])}")
print(f"Spada Reservoir: {profiles['spada_reservoir']['current_elevation_ft']} / {profiles['spada_reservoir']['max_level_ft']} ft")
print(f"YTD Precip: {profiles['precipitation']['ytd_inches']}\" ({profiles['precipitation']['ratio_to_normal']:.0%} of normal)")
