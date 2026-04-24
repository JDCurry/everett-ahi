"""
Everett AHI — City-Level Hazard Dashboard
==========================================
Adaptive Hazard Intelligence for the City of Everett
Office of Emergency Management.

Run: streamlit run dashboard/app.py
"""

import json
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

PLATFORM_NAME = "Adaptive Hazard Intelligence"
TAGLINE = "City of Everett — Calibrated hazard risk for defensible decisions"

# Resolve paths using absolute paths
_APP_DIR = Path(__file__).resolve().parent
DATA_DIR = _APP_DIR.parent / "data" / "processed"
EVERETT_DATA = Path("everett_data")

# Verify DATA_DIR exists
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
if not (DATA_DIR / "neighborhood_profiles.json").exists():
    raise FileNotFoundError(f"neighborhood_profiles.json not found at: {DATA_DIR / 'neighborhood_profiles.json'}")

HAZARDS = ["fire", "flood", "wind", "winter", "seismic"]
HAZARD_LABELS = {
    "fire": "Wildfire",
    "flood": "Flood",
    "wind": "Severe Wind",
    "winter": "Winter Storm",
    "seismic": "Earthquake",
}

COLORS = {
    "app_bg": "#0f1419",
    "card_bg": "#1a1f26",
    "sidebar_bg": "#141920",
    "elevated_bg": "#1f2430",
    "primary": "#0d7dc1",
    "critical": "#c0392b",
    "warning": "#e8a838",
    "success": "#27ae60",
    "info": "#3498db",
    "border": "#2d333b",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "text_tertiary": "#6e7681",
    "fire": "#c0392b",
    "flood": "#2980b9",
    "wind": "#8e44ad",
    "winter": "#1abc9c",
    "seismic": "#d35400",
}

# Human-readable weather variable names
WEATHER_LABELS = {
    "tmmx": "Max Temperature",
    "tmmn": "Min Temperature",
    "pr": "Precipitation",
    "vs": "Wind Speed",
    "vpd": "Vapor Pressure Deficit",
    "erc": "Energy Release Component",
    "rmin": "Min Relative Humidity",
    "rmax": "Max Relative Humidity",
}

# Neighborhood map: reframe "calls" as "incidents"
NBR_MAP_METRICS = {
    "incident_rate": "Incident Rate (per 100 acres)",
    "total_incidents": "Total EM Incidents",
    "num_basins": "Flood Basin Exposure",
    "fire_incidents": "Fire Incidents",
    "winter_incidents": "Winter Incidents",
}


def risk_level(index_val):
    if index_val >= 75:
        return "High", COLORS["critical"]
    elif index_val >= 50:
        return "Elevated", COLORS["warning"]
    elif index_val >= 25:
        return "Moderate", COLORS["info"]
    return "Low", COLORS["success"]


def get_plotly_dark():
    return dict(
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text_secondary"], family="Segoe UI"),
    )


# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown(f"""<style>
    .stApp {{
        background: {COLORS['app_bg']} !important;
        color: {COLORS['text_secondary']} !important;
    }}
    h1 {{
        color: {COLORS['text_primary']} !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        border-bottom: 2px solid {COLORS['primary']} !important;
        padding-bottom: 12px;
    }}
    h2, h3 {{
        color: {COLORS['text_primary']} !important;
        font-family: 'Segoe UI', sans-serif;
    }}
    section[data-testid="stSidebar"] {{
        background: {COLORS['sidebar_bg']} !important;
        border-right: 1px solid {COLORS['border']} !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {{
        color: {COLORS['text_primary']} !important;
    }}
    [data-testid="metric-container"] {{
        background: {COLORS['card_bg']} !important;
        border: 1px solid {COLORS['border']} !important;
        padding: 18px;
        border-radius: 6px;
    }}
    [data-testid="metric-container"] label {{
        color: {COLORS['text_secondary']} !important;
        text-transform: uppercase;
        font-size: 11px;
    }}
    [data-testid="metric-container"] [data-testid="metric-value"] {{
        color: {COLORS['text_primary']} !important;
    }}
    .stButton > button {{
        background-color: {COLORS['primary']} !important;
        color: white !important;
        border: none;
        border-radius: 4px;
    }}
    .stat-card {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 16px;
    }}
    .alert-box {{
        padding: 16px 20px;
        border-radius: 6px;
        margin: 12px 0;
        border-left: 4px solid;
    }}
    .alert-critical {{
        background: rgba(192, 57, 43, 0.1) !important;
        border-left-color: {COLORS['critical']} !important;
    }}
    .alert-warning {{
        background: rgba(232, 168, 56, 0.1) !important;
        border-left-color: {COLORS['warning']} !important;
    }}
    .alert-info {{
        background: rgba(13, 125, 193, 0.1) !important;
        border-left-color: {COLORS['primary']} !important;
    }}
    .alert-success {{
        background: rgba(39, 174, 96, 0.1) !important;
        border-left-color: {COLORS['success']} !important;
    }}
    .hazard-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
        color: white;
    }}
    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding: 10px 0 20px 0;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 20px;
    }}
    </style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    profiles = json.loads((DATA_DIR / "neighborhood_profiles.json").read_text())
    predictions = pd.read_parquet(DATA_DIR / "everett_predictions.parquet")
    predictions["date"] = pd.to_datetime(predictions["date"])
    neighborhoods = gpd.read_file(EVERETT_DATA / "Neighborhoods_20260413.geojson")
    neighborhoods = neighborhoods.rename(columns={"name": "neighborhood"})
    neighborhoods["neighborhood"] = neighborhoods["neighborhood"].replace(
        {"PINEHURST": "PINEHURST-BEVERLY PARK"}
    )
    return profiles, predictions, neighborhoods


# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

def render_header():
    try:
        from zoneinfo import ZoneInfo
        pst = ZoneInfo("America/Los_Angeles")
        now = datetime.now(pst)
    except Exception:
        from datetime import timezone
        pst = timezone(timedelta(hours=-8))
        now = datetime.now(pst)

    st.markdown(f"""
    <div class="header-container">
        <div>
            <h1 style="border: none !important; margin: 0; padding: 0; font-size: 28px;">
                {PLATFORM_NAME}
            </h1>
            <p style="color: {COLORS['text_tertiary']}; font-size: 13px; margin: 4px 0 0 0;">
                {TAGLINE}
            </p>
        </div>
        <div style="text-align: right;">
            <div style="color: {COLORS['text_primary']}; font-size: 18px; font-weight: 600;">
                {now.strftime('%H:%M:%S')}
            </div>
            <div style="color: {COLORS['text_secondary']}; font-size: 12px;">
                {now.strftime('%B %d, %Y')}
            </div>
            <div style="color: {COLORS['success']}; font-size: 11px; font-weight: 600; margin-top: 8px;">
                AHI v2.5 — Learned Seasonal Bias
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════

def page_risk_overview(predictions, profiles, forecast_date, selected_hazard):
    """Risk assessment with forecast horizon."""

    # Use seasonal pattern from historical data for forecast dates
    # Map forecast date to same month/day in the most recent available year
    forecast_dt = pd.Timestamp(forecast_date)
    month = forecast_dt.month
    day_of_year = forecast_dt.timetuple().tm_yday

    # Find closest matching day in historical predictions
    predictions["doy"] = predictions["date"].dt.dayofyear
    candidates = predictions[predictions["doy"] == day_of_year]
    if len(candidates) == 0:
        candidates = predictions[(predictions["doy"] >= day_of_year - 3) &
                                  (predictions["doy"] <= day_of_year + 3)]
    if len(candidates) == 0:
        candidates = predictions[predictions["date"].dt.month == month]

    # Average across all matching historical days for the forecast
    row = candidates[[f"{h}_risk_index" for h in HAZARDS] +
                      [f"{h}_city" for h in HAZARDS] +
                      [f"{h}_raw" for h in HAZARDS] +
                      ["tmmx", "tmmn", "pr", "vs"]].mean()

    # Risk cards
    cols = st.columns(5)
    for i, h in enumerate(HAZARDS):
        idx = row[f"{h}_risk_index"]
        level, color = risk_level(idx)
        with cols[i]:
            st.markdown(f"""
            <div class="stat-card" style="border-left: 4px solid {COLORS[h]};">
                <div style="color: {COLORS['text_tertiary']}; font-size: 11px; text-transform: uppercase;">
                    {HAZARD_LABELS[h]}
                </div>
                <div style="color: {color}; font-size: 28px; font-weight: 600;">
                    {idx:.0f}
                </div>
                <div style="color: {color}; font-size: 12px; font-weight: 600;">
                    {level}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Weather conditions for forecast date
    st.subheader("Forecast Conditions")
    precip = profiles["precipitation"]
    wcols = st.columns(5)
    weather_display = [
        ("tmmx", "Max Temperature"),
        ("tmmn", "Min Temperature"),
        ("pr", "Precipitation"),
        ("vs", "Wind Speed"),
    ]
    for i, (col_name, label) in enumerate(weather_display):
        val = row.get(col_name, 0)
        if pd.notna(val) and col_name in ("tmmx", "tmmn"):
            val_f = (val - 273.15) * 9/5 + 32
            wcols[i].metric(label, f"{val_f:.0f}°F")
        elif pd.notna(val) and col_name == "pr":
            val_in = val / 25.4  # mm to inches
            wcols[i].metric(label, f'{val_in:.2f}"')
        elif pd.notna(val) and col_name == "vs":
            mph = val * 2.237
            wcols[i].metric(label, f"{mph:.0f} mph")
    # YTD precipitation alongside forecast weather
    wcols[4].metric(
        "YTD Precipitation",
        f'{precip["ytd_inches"]:.1f}"',
        delta=f'{precip["ratio_to_normal"]:.0%} of normal',
        delta_color="off",
    )

    st.divider()

    # Operational trigger for selected hazard
    focus_idx = row[f"{selected_hazard}_risk_index"]
    focus_level, focus_color = risk_level(focus_idx)

    trigger_text = {
        "fire": {
            "Low": "No action required. Continue routine monitoring.",
            "Moderate": "Verify fire station readiness. Monitor red flag warnings from NWS.",
            "Elevated": "Pre-position brush units. Consider public advisory for outdoor burning restrictions.",
            "High": "Activate fire mutual aid agreements. Issue burn ban. Brief EOC staff.",
        },
        "flood": {
            "Low": "No action required. Monitor Spada Reservoir levels.",
            "Moderate": "Check pump station readiness. Monitor river gauge levels and NWS flood watches.",
            "Elevated": "Pre-stage sandbags at identified flood-prone locations. Alert public works.",
            "High": "Activate flood annex. Open EOC. Coordinate with Snohomish County flood control.",
        },
        "wind": {
            "Low": "No action required.",
            "Moderate": "Monitor NWS wind advisories. Verify tree-trimming schedule with public works.",
            "Elevated": "Pre-position utility crews. Coordinate with Snohomish PUD for outage response.",
            "High": "Activate severe weather annex. Shelter assessment. Power outage coordination.",
        },
        "winter": {
            "Low": "No action required.",
            "Moderate": "Pre-treat priority routes. Check warming shelter capacity at identified parks.",
            "Elevated": "Activate snow/ice plan. Open warming shelters. Coordinate transit adjustments.",
            "High": "Full winter storm response. EOC activation. Mutual aid requests.",
        },
        "seismic": {
            "Low": "Maintain standard preparedness posture.",
            "Moderate": "Maintain standard preparedness posture.",
            "Elevated": "Review earthquake annex. Verify CERT team readiness.",
            "High": "Automatic EOC activation per CEMP. Initiate damage assessment protocol.",
        },
    }

    alert_class = {
        "Low": "alert-success", "Moderate": "alert-info",
        "Elevated": "alert-warning", "High": "alert-critical",
    }[focus_level]

    action = trigger_text.get(selected_hazard, {}).get(focus_level, "Monitor conditions.")
    st.markdown(f"""
    <div class="alert-box {alert_class}">
        <div style="font-size: 14px; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 8px;">
            {HAZARD_LABELS[selected_hazard]} — {focus_level} (Risk Index: {focus_idx:.0f}/100)
        </div>
        <div style="color: {COLORS['text_secondary']}; font-size: 13px;">
            {action}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CPRI Rankings
    st.subheader("CPRI Risk Rankings")
    st.caption("2024 FEMA-Approved Hazard Mitigation Plan")
    cpri = profiles["cpri_rankings"]
    cpri_df = pd.DataFrame([
        {"Hazard": k.replace("_", " ").title(), "Score": v["score"],
         "Rank": v["rank"], "Tier": v["tier"]}
        for k, v in cpri.items()
    ]).sort_values("Rank")
    st.dataframe(cpri_df, use_container_width=True, hide_index=True)


def page_neighborhood_map(profiles, neighborhoods):
    """Neighborhood choropleth with clean hover labels."""

    nbr_data = []
    for name, p in profiles["neighborhoods"].items():
        flood_exp = p.get("flood_exposure", {})
        fire = p["hazard_call_counts"].get("fire", 0)
        wildfire = p["hazard_call_counts"].get("wildfire", 0)
        winter = p["hazard_call_counts"].get("winter", 0)
        nbr_data.append({
            "neighborhood": name,
            "total_incidents": p["total_hazard_calls"],
            "fire_incidents": fire + wildfire,
            "winter_incidents": winter,
            "area_acres": p["area_acres"],
            "num_basins": flood_exp.get("num_basins", 0),
            "has_levee": flood_exp.get("has_levee_protection", False),
            "basins": ", ".join(flood_exp.get("drainage_basins", [])),
        })

    nbr_df = pd.DataFrame(nbr_data)
    nbr_df["incident_rate"] = (nbr_df["total_incidents"] / nbr_df["area_acres"].replace(0, 1) * 100).round(1)

    gdf = neighborhoods.merge(nbr_df, on="neighborhood", how="left")

    # Clean display names for neighborhoods
    gdf["display_name"] = gdf["neighborhood"].str.title()

    map_metric = st.selectbox(
        "Color by",
        list(NBR_MAP_METRICS.keys()),
        format_func=lambda x: NBR_MAP_METRICS[x],
    )

    color_scale = {
        "incident_rate": "YlOrRd",
        "total_incidents": "YlOrRd",
        "num_basins": "Blues",
        "fire_incidents": "Reds",
        "winter_incidents": "BuPu",
    }[map_metric]

    # Build custom hover text
    hover_texts = []
    for _, row in gdf.iterrows():
        levee_str = "Yes" if row.get("has_levee", False) else "No"
        hover_texts.append(
            f"<b>{row['display_name']}</b><br>"
            f"EM Incidents: {row['total_incidents']:,.0f}<br>"
            f"Incident Rate: {row['incident_rate']:.1f} per 100 ac<br>"
            f"Area: {row['area_acres']:,.0f} acres<br>"
            f"Flood Basins: {row['num_basins']}<br>"
            f"Levee Protected: {levee_str}"
        )

    fig = go.Figure(go.Choroplethmapbox(
        geojson=gdf.geometry.__geo_interface__,
        locations=gdf.index,
        z=gdf[map_metric],
        colorscale=color_scale,
        text=hover_texts,
        hoverinfo="text",
        marker_opacity=0.7,
        marker_line_width=1.5,
        marker_line_color=COLORS["border"],
        colorbar=dict(
            title=dict(text=NBR_MAP_METRICS[map_metric], font=dict(size=12)),
            tickfont=dict(size=10),
        ),
    ))
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_center={"lat": 47.979, "lon": -122.202},
        mapbox_zoom=11.5,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
        **get_plotly_dark(),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Neighborhood table
    st.subheader("Neighborhood Details")
    display_df = nbr_df.sort_values("incident_rate", ascending=False)[
        ["neighborhood", "total_incidents", "incident_rate", "area_acres",
         "fire_incidents", "winter_incidents", "num_basins", "has_levee"]
    ].copy()
    display_df["neighborhood"] = display_df["neighborhood"].str.title()
    display_df = display_df.rename(columns={
        "neighborhood": "Neighborhood",
        "total_incidents": "EM Incidents",
        "incident_rate": "Rate/100ac",
        "area_acres": "Area (ac)",
        "fire_incidents": "Fire",
        "winter_incidents": "Winter",
        "num_basins": "Flood Basins",
        "has_levee": "Levee Protected",
    })
    st.dataframe(
        display_df.style.format({
            "EM Incidents": "{:,.0f}",
            "Rate/100ac": "{:.1f}",
            "Area (ac)": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # Shelter inventory
    st.subheader("Emergency Shelter & Gathering Facilities")
    st.caption("Parks with covered shelter or gazebo structures (source: data.everettwa.gov)")
    shelters = profiles.get("shelters", [])
    if shelters:
        shelter_df = pd.DataFrame(shelters).rename(columns={
            "name": "Facility", "address": "Address", "acreage": "Acreage",
        })
        st.dataframe(shelter_df, use_container_width=True, hide_index=True)


def page_historical_trends(predictions, selected_hazard):
    """Historical risk trends with dark-themed charts."""

    # Cap to most recent 5 years
    max_year = predictions["date"].dt.year.max()
    filtered = predictions[
        predictions["date"].dt.year >= max_year - 4
    ].copy()

    # Time series
    fig = go.Figure()
    for h in HAZARDS:
        col = f"{h}_risk_index"
        smoothed = filtered.set_index("date")[col].rolling(30, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=smoothed.index, y=smoothed.values,
            name=HAZARD_LABELS[h],
            line=dict(color=COLORS[h], width=2),
        ))

    fig.update_layout(
        title="30-Day Rolling Risk Index",
        yaxis_title="Risk Index (0–100)",
        xaxis_title="",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **get_plotly_dark(),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.subheader("Seasonal Risk Patterns")
    hazard_to_show = st.selectbox(
        "Hazard", HAZARDS,
        format_func=lambda h: HAZARD_LABELS[h],
        key="heatmap_hazard",
    )

    monthly = filtered.groupby([
        filtered["date"].dt.year.rename("year"),
        filtered["date"].dt.month.rename("month"),
    ])[[f"{h}_risk_index" for h in HAZARDS]].mean()

    pivot = monthly[f"{hazard_to_show}_risk_index"].reset_index().pivot(
        index="year", columns="month", values=f"{hazard_to_show}_risk_index"
    )
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig_hm = px.imshow(
        pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        color_continuous_scale="YlOrRd",
        labels=dict(x="Month", y="Year", color="Risk Index"),
        title=f"{HAZARD_LABELS[hazard_to_show]} — Year × Month",
    )
    fig_hm.update_layout(height=400, **get_plotly_dark())
    st.plotly_chart(fig_hm, use_container_width=True)

    # Weather correlation
    st.subheader("Weather–Risk Relationship")
    weather_var = st.selectbox(
        "Weather Variable",
        ["tmmx", "pr", "vs", "vpd", "erc"],
        format_func=lambda x: WEATHER_LABELS.get(x, x),
    )
    if weather_var in filtered.columns:
        sample = filtered.sample(min(2000, len(filtered)), random_state=42)
        # Convert units for display
        x_data = sample[weather_var].copy()
        x_label = WEATHER_LABELS.get(weather_var, weather_var)
        if weather_var in ("tmmx", "tmmn"):
            x_data = (x_data - 273.15) * 9/5 + 32
            x_label += " (°F)"
        elif weather_var == "pr":
            x_data = x_data / 25.4  # mm to inches
            x_label += ' (in)'
        elif weather_var == "vs":
            x_data = x_data * 2.237
            x_label += " (mph)"
        elif weather_var == "vpd":
            x_label += " (kPa)"

        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_corr = go.Figure(go.Scatter(
            x=x_data, y=sample[f"{selected_hazard}_risk_index"],
            mode="markers",
            marker=dict(
                color=sample["month"],
                colorscale="Viridis",
                cmin=1, cmax=12,
                size=4,
                opacity=0.4,
                colorbar=dict(
                    title="Month",
                    tickvals=list(range(1, 13)),
                    ticktext=month_names,
                ),
            ),
        ))
        fig_corr.update_layout(
            title=f"{HAZARD_LABELS[selected_hazard]} Risk vs {WEATHER_LABELS.get(weather_var, weather_var)}",
            xaxis_title=x_label,
            yaxis_title="Risk Index",
            height=400,
            **get_plotly_dark(),
        )
        st.plotly_chart(fig_corr, use_container_width=True)


def page_about():
    """About page with data source attribution."""
    st.markdown(f"""
### Model

**AHI v2.5** — Learned Seasonal Bias (Experiment D)

| Metric | Value |
|--------|-------|
| Architecture | HazardLMDiffusion with heat kernel attention |
| Parameters | 1,294,547 |
| Mean Test AUC | 0.829 |
| Fire AUC | 0.851 |
| Flood AUC | 0.830 |
| Severe Wind AUC | 0.837 |
| Winter Storm AUC | 0.908 |
| Earthquake AUC | 0.718 |

The model predicts daily probability of five hazard types based on weather conditions,
geographic features, and learned seasonal patterns. For this city-level prototype,
county-level weather inputs are replaced with data from Everett's actual
[GridMET](https://www.climatologylab.org/gridmet.html) grid cell (47.98°N, 122.20°W)
rather than the Snohomish County centroid near Monroe.

### Risk Index

The Risk Index (0–100) represents **relative risk** compared to Everett's own
25-year historical range. An index of 75 means conditions are in the top 25% of
historical risk days for that hazard — not a 75% probability of an event occurring.

| Level | Index Range | Color |
|-------|-------------|-------|
| Low | 0–24 | Green |
| Moderate | 25–49 | Blue |
| Elevated | 50–74 | Orange |
| High | 75–100 | Red |

Operational trigger language is proposed and requires validation by the OEM director
before operational use.

---

### Data Sources

**Weather & Hazard Model**
- [GridMET](https://www.climatologylab.org/gridmet.html) — Daily 4km meteorological
  data (2000–present). Variables: temperature, precipitation, wind speed, humidity,
  vapor pressure deficit, energy release component.
- [NOAA Storm Events](https://www.ncdc.noaa.gov/stormevents/) — Historical hazard event records
- [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/) — Seismic event data
- [NWS Forecast API](https://www.weather.gov/documentation/services-web-api) — 2.5km grid forecasts

**City of Everett Open Data** — [data.everettwa.gov](https://data.everettwa.gov)
- Neighborhood boundaries (19 neighborhoods)
- Neighborhood–Census Tract Bridge (96 mappings)
- Fire 911 Unit Dispatches (87,256 records; EM-relevant incidents classified)
- Drainage Basins (27 basins, spatial overlay with neighborhoods)
- Flood Levee infrastructure (49 segments; accreditation status)
- Spada Reservoir elevation and cumulative precipitation
- Parks and Amenities (shelter/gathering point identification)
- Building Footprint (58,956 structures)

**Hazard Mitigation Plan**
- 2024 FEMA-Approved Snohomish County Hazard Mitigation Plan (Everett annex)
- CPRI Rankings: Earthquake 3.85, Flood 2.85, Severe Weather 2.85,
  Landslide 2.35, Wildfire 2.30

---

### Limitations

- **Prototype status** — Uses an existing county-level model with swapped city-level
  weather inputs. A model retrained on city-specific labels would improve accuracy.
- **Earthquake** — Modeled as constant geographic background risk. The seismic head
  (AUC 0.72) is the weakest and should not be used for earthquake prediction.
- **Landslide** — Ranked #4 in Everett's CPRI but not in the current AHI feature set.
- **Forecast horizon** — Extended forecasts (14–30 days) use seasonal averages rather
  than day-specific weather predictions. Shorter horizons are more reliable.

---

### Contact

Developed by Joshua D. Curry — Pierce College BAS Emergency Management
capstone project and DHS SBIR Phase I preparation.

**Resilience Analytics Lab, LLC**
    """)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title=PLATFORM_NAME,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()
    profiles, predictions, neighborhoods = load_data()

    # ── Sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        logo_path = Path(__file__).parent / "logo.png"
        if logo_path.exists():
            import base64
            logo_bytes = logo_path.read_bytes()
            logo_b64 = base64.b64encode(logo_bytes).decode()
            st.markdown(f"""
            <div style='text-align: center; padding: 15px 0 10px 0; border-bottom: 1px solid {COLORS["border"]};'>
                <img src='data:image/png;base64,{logo_b64}' width='60' style='margin-bottom: 8px;'>
                <h2 style='color: {COLORS["text_primary"]}; margin: 0; font-size: 16px;'>
                    Everett AHI
                </h2>
                <p style='color: {COLORS["text_tertiary"]}; font-size: 11px; margin: 2px 0 0 0;'>
                    City-Level Hazard Intelligence
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align: center; padding: 15px 0 10px 0; border-bottom: 1px solid {COLORS["border"]};'>
                <h2 style='color: {COLORS["text_primary"]}; margin: 0; font-size: 16px;'>
                    Everett AHI
                </h2>
                <p style='color: {COLORS["text_tertiary"]}; font-size: 11px; margin: 2px 0 0 0;'>
                    City-Level Hazard Intelligence
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Forecast horizon
        st.markdown(f"<p style='color: {COLORS['text_tertiary']}; font-size: 14px; font-weight: 600; "
                    f"text-transform: uppercase; margin-top: 20px; letter-spacing: 0.5px;'>"
                    f"Forecast</p>", unsafe_allow_html=True)

        forecast_choice = st.selectbox(
            "Forecast Horizon",
            ["7 days", "14 days", "30 days (extended)"],
            index=1,
            help="Shorter horizon preferred for reliability; extended window carries greater uncertainty.",
        )

        max_days = int(forecast_choice.split()[0])

        try:
            from zoneinfo import ZoneInfo
            today = datetime.now(ZoneInfo("America/Los_Angeles")).date()
        except Exception:
            from datetime import timezone
            today = datetime.now(timezone(timedelta(hours=-8))).date()

        max_forecast = today + timedelta(days=max_days)
        forecast_date = st.date_input(
            "Forecast Date",
            value=today,
            min_value=today,
            max_value=max_forecast,
        )

        # Hazard selector
        st.markdown(f"<p style='color: {COLORS['text_tertiary']}; font-size: 14px; font-weight: 600; "
                    f"text-transform: uppercase; margin-top: 20px; letter-spacing: 0.5px;'>"
                    f"Focus Hazard</p>", unsafe_allow_html=True)

        selected_hazard = st.selectbox(
            "Hazard",
            HAZARDS,
            format_func=lambda h: HAZARD_LABELS[h],
            label_visibility="collapsed",
        )

        # Navigation
        st.markdown(f"<p style='color: {COLORS['text_tertiary']}; font-size: 14px; font-weight: 600; "
                    f"text-transform: uppercase; margin-top: 20px; letter-spacing: 0.5px;'>"
                    f"Modules</p>", unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state.page = "overview"

        if st.button("Risk Overview", use_container_width=True):
            st.session_state.page = "overview"
        if st.button("Neighborhood Map", use_container_width=True):
            st.session_state.page = "map"
        if st.button("Historical Trends", use_container_width=True):
            st.session_state.page = "trends"
        if st.button("About & Data Sources", use_container_width=True):
            st.session_state.page = "about"

    # ── Header ──────────────────────────────────────────────────────────
    render_header()

    # ── Page routing ────────────────────────────────────────────────────
    page = st.session_state.get("page", "overview")

    if page == "overview":
        page_risk_overview(predictions, profiles, forecast_date, selected_hazard)
    elif page == "map":
        page_neighborhood_map(profiles, neighborhoods)
    elif page == "trends":
        page_historical_trends(predictions, selected_hazard)
    elif page == "about":
        page_about()


if __name__ == "__main__":
    main()
