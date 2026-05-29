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
import warnings
from scipy.stats import gamma

warnings.filterwarnings('ignore')

# =============================================================================
# PROBABILISTIC RISK SCORING MODULE (Gamma-Poisson Empirical Bayes)
# =============================================================================

def gamma_poisson_posterior(events, days_observed, prior_alpha=0.5, prior_beta=0.5, threshold=None, ci=0.90):
    """
    Empirical Bayes Gamma-Poisson posterior for incident rate estimation.
    """
    events = np.asarray(events)
    days_observed = np.asarray(days_observed)
    safe_days = np.where(days_observed > 0, days_observed, 1)
    post_alpha = prior_alpha + events
    post_beta = prior_beta + safe_days
    mean_rate_per_day = post_alpha / post_beta
    posterior_mean_rate = mean_rate_per_day * 365.0
    lower = gamma.ppf((1 - ci) / 2, post_alpha, scale=1 / post_beta)
    upper = gamma.ppf(1 - (1 - ci) / 2, post_alpha, scale=1 / post_beta)
    lower_ci = lower * 365.0
    upper_ci = upper * 365.0
    prob_exceeds = None
    if threshold is not None:
        thresh_per_day = threshold / 365.0
        prob_exceeds = 1 - gamma.cdf(thresh_per_day, post_alpha, scale=1 / post_beta)
    return posterior_mean_rate, lower_ci, upper_ci, prob_exceeds

# =============================================================================

PLATFORM_NAME = "Adaptive Hazard Intelligence"
TAGLINE = "City of Everett — Calibrated hazard risk for defensible decisions"

# Resolve paths using absolute paths
_APP_DIR = Path(__file__).resolve().parent
DATA_DIR = _APP_DIR.parent / "data" / "processed"
EVERETT_DATA = DATA_DIR

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

# Neighborhood map metrics
NBR_MAP_METRICS = {
    "incident_rate": "Response Rate (per 100 acres)",
    "total_incidents": "Total Response Calls",
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


def inject_css_theme():
    """Return plotly theme dict for charts."""
    return {
        'paper_bgcolor': COLORS['card_bg'],
        'plot_bgcolor': COLORS['card_bg'],
        'font': {'color': COLORS['text_secondary']},
        'xaxis': {'gridcolor': COLORS['border'], 'linecolor': COLORS['border']},
        'yaxis': {'gridcolor': COLORS['border'], 'linecolor': COLORS['border']},
    }


# =====================================================================
# CSS
# =====================================================================

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


# =====================================================================
# DATA LOADING
# =====================================================================

def compute_neighborhood_adjustments(profiles):
    """Compute neighborhood-specific adjustment factors based on hazard call counts."""
    neighborhoods = profiles["neighborhoods"]

    total_calls = {}
    total_area = 0.0
    for nbr, prof in neighborhoods.items():
        total_area += prof.get("area_acres", 0)
        calls = prof.get("hazard_call_counts", {})
        for hazard_type, count in calls.items():
            total_calls[hazard_type] = total_calls.get(hazard_type, 0) + count

    city_rates = {h: (count / total_area if total_area > 0 else 0) for h, count in total_calls.items()}

    adjustments = {}
    for nbr, prof in neighborhoods.items():
        area = prof.get("area_acres", 1.0)
        calls = prof.get("hazard_call_counts", {})
        nbr_adjustments = {}
        for h in HAZARDS:
            nbr_rate = calls.get(h, 0) / area if area > 0 else 0
            city_rate = city_rates.get(h, 0.001)
            factor = nbr_rate / city_rate if city_rate > 0 else 1.0
            nbr_adjustments[h] = max(0.5, min(2.0, factor))
        adjustments[nbr] = nbr_adjustments

    return adjustments


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

    # Cross-join city-level predictions with all neighborhoods
    nbr_list = sorted(neighborhoods["neighborhood"].unique().tolist())
    predictions = predictions.copy()
    predictions["key"] = 1
    nbr_df = pd.DataFrame({"neighborhood": nbr_list, "key": 1})
    predictions = predictions.merge(nbr_df, on="key", how="left").drop("key", axis=1)

    return profiles, predictions, neighborhoods


def compute_neighborhood_stats(predictions, profiles):
    """Compute neighborhood-level risk statistics using Gamma-Poisson posterior."""
    nbr_stats = {}
    neighborhoods = profiles["neighborhoods"]
    adjustments = compute_neighborhood_adjustments(profiles)

    for nbr, prof in neighborhoods.items():
        nbr_df = predictions[predictions["neighborhood"] == nbr].copy()
        days_observed = len(nbr_df)

        if days_observed == 0:
            nbr_stats[nbr] = {
                "fire_rate": 0.0, "flood_rate": 0.0,
                "fire_events": 0, "flood_events": 0
            }
            continue

        nbr_adjustments = adjustments.get(nbr, {h: 1.0 for h in HAZARDS})
        nbr_df["fire_risk_index"] = (nbr_df["fire_risk_index"] * nbr_adjustments.get("fire", 1.0)).clip(0, 100)
        nbr_df["flood_risk_index"] = (nbr_df["flood_risk_index"] * nbr_adjustments.get("flood", 1.0)).clip(0, 100)

        fire_events = (nbr_df["fire_risk_index"] > 50).sum()
        flood_events = (nbr_df["flood_risk_index"] > 50).sum()

        fire_mean, fire_lower, fire_upper, _ = gamma_poisson_posterior(
            fire_events, days_observed, prior_alpha=0.5, prior_beta=0.5, ci=0.90
        )
        flood_mean, flood_lower, flood_upper, _ = gamma_poisson_posterior(
            flood_events, days_observed, prior_alpha=0.5, prior_beta=0.5, ci=0.90
        )

        nbr_stats[nbr] = {
            "fire_rate": fire_mean,
            "flood_rate": flood_mean,
            "fire_events": int(fire_events),
            "flood_events": int(flood_events),
            "days_observed": days_observed,
            "fire_lower_ci": fire_lower, "fire_upper_ci": fire_upper,
            "flood_lower_ci": flood_lower, "flood_upper_ci": flood_upper,
        }

    return nbr_stats


# =====================================================================
# HEADER
# =====================================================================

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
                AHI Round 3 — PNW Regional Model
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# PAGES
# =====================================================================

def page_risk_overview(predictions, profiles, selected_hazard):
    """Risk assessment for current conditions."""

    try:
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    except Exception:
        from datetime import timezone
        today = datetime.now(timezone(timedelta(hours=-8))).date()

    forecast_dt = pd.Timestamp(today)
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

    # Average across all matching historical days
    row = candidates[[f"{h}_risk_index" for h in HAZARDS] +
                      [f"{h}_city" for h in HAZARDS] +
                      [f"{h}_raw" for h in HAZARDS] +
                      ["tmmx", "tmmn", "pr", "vs"]].mean()

    # Risk cards (seismic excluded from display)
    display_hazards = [h for h in HAZARDS if h != "seismic"]
    cols = st.columns(len(display_hazards))
    for i, h in enumerate(display_hazards):
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

    # Weather conditions
    st.subheader("Current Conditions")
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
            wcols[i].metric(label, f"{val_f:.0f} F")
        elif pd.notna(val) and col_name == "pr":
            val_in = val / 25.4
            wcols[i].metric(label, f'{val_in:.2f}"')
        elif pd.notna(val) and col_name == "vs":
            mph = val * 2.237
            wcols[i].metric(label, f"{mph:.0f} mph")
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

    # Build hover text
    hover_texts = []
    for _, row in gdf.iterrows():
        levee_str = "Yes" if row.get("has_levee", False) else "No"
        hover_texts.append(
            f"<b>{row['display_name']}</b><br>"
            f"EM Incidents: {row['total_incidents']:,.0f}<br>"
            f"Rate: {row['incident_rate']:.1f}/100ac<br>"
            f"Area: {row['area_acres']:,.0f} acres<br>"
            f"Flood Basins: {row['num_basins']}<br>"
            f"Levee: {levee_str}"
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
            title=dict(text=NBR_MAP_METRICS[map_metric], font=dict(size=11), side="bottom"),
            tickfont=dict(size=10),
            x=1.0,
            xpad=10,
            len=0.85,
            yanchor="middle",
            y=0.5,
        ),
    ))
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_center={"lat": 47.979, "lon": -122.202},
        mapbox_zoom=10.8,
        margin={"r": 80, "t": 0, "l": 0, "b": 30},
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

    # Time series (seismic excluded from display)
    trend_hazards = [h for h in HAZARDS if h != "seismic"]
    fig = go.Figure()
    for h in trend_hazards:
        col = f"{h}_risk_index"
        smoothed = filtered.set_index("date")[col].rolling(30, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=smoothed.index, y=smoothed.values,
            name=HAZARD_LABELS[h],
            line=dict(color=COLORS[h], width=2),
        ))

    fig.update_layout(
        title="30-Day Rolling Risk Index",
        yaxis_title="Risk Index (0-100)",
        xaxis_title="",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **get_plotly_dark(),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap (seismic excluded — no seasonal pattern, not actionable)
    st.subheader("Seasonal Risk Patterns")
    display_hazards = [h for h in HAZARDS if h != "seismic"]
    hazard_to_show = st.selectbox(
        "Hazard", display_hazards,
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
        title=f"{HAZARD_LABELS[hazard_to_show]} — Year x Month",
    )
    fig_hm.update_layout(height=400, **get_plotly_dark())
    st.plotly_chart(fig_hm, use_container_width=True)

    # Weather correlation
    st.subheader("Weather-Risk Relationship")
    weather_var = st.selectbox(
        "Weather Variable",
        ["tmmx", "pr", "vs", "vpd", "erc"],
        format_func=lambda x: WEATHER_LABELS.get(x, x),
    )
    if weather_var in filtered.columns:
        sample = filtered.sample(min(2000, len(filtered)), random_state=42)
        x_data = sample[weather_var].copy()
        x_label = WEATHER_LABELS.get(weather_var, weather_var)
        if weather_var in ("tmmx", "tmmn"):
            x_data = (x_data - 273.15) * 9/5 + 32
            x_label += " (F)"
        elif weather_var == "pr":
            x_data = x_data / 25.4
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


def page_quick_predict(predictions, profiles, neighborhoods, selected_hazard):
    """Quick Predict: Neighborhood-level hazard forecast with inline controls."""
    st.markdown("## Quick Predict")
    st.markdown("Select a neighborhood and forecast window to generate hazard predictions.")

    # ── Inline forecast controls ────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])
    with ctrl_col1:
        nbr_names = sorted(profiles["neighborhoods"].keys())
        selected_nbr = st.selectbox("Neighborhood", nbr_names, format_func=lambda n: n.title())
    with ctrl_col2:
        forecast_choice = st.selectbox(
            "Forecast Horizon",
            ["14 days", "30 days"],
            index=0,
            help="14-day forecasts use recent seasonal patterns. 30-day forecasts use monthly averages.",
        )
    with ctrl_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        calculate = st.button("Calculate", use_container_width=True, type="primary")

    horizon_days = int(forecast_choice.split()[0])

    # Get neighborhood data
    nbr_df = predictions[predictions["neighborhood"] == selected_nbr].sort_values("date", ascending=False)

    if len(nbr_df) == 0:
        st.warning(f"No prediction data available for {selected_nbr.title()}.")
        return

    # For the forecast, average over the horizon window centered on today's DOY
    try:
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    except Exception:
        from datetime import timezone
        today = datetime.now(timezone(timedelta(hours=-8))).date()

    today_doy = today.timetuple().tm_yday
    nbr_df = nbr_df.copy()
    nbr_df["doy"] = nbr_df["date"].dt.dayofyear

    # Select matching days within the forecast horizon window
    end_doy = today_doy + horizon_days
    if end_doy <= 365:
        window_mask = (nbr_df["doy"] >= today_doy) & (nbr_df["doy"] <= end_doy)
    else:
        # Wrap around year boundary
        window_mask = (nbr_df["doy"] >= today_doy) | (nbr_df["doy"] <= end_doy - 365)

    forecast_data = nbr_df[window_mask]
    if len(forecast_data) == 0:
        forecast_data = nbr_df.head(100)  # Fallback

    # Compute neighborhood-specific adjustments
    adjustments = compute_neighborhood_adjustments(profiles)
    nbr_adjustments = adjustments.get(selected_nbr, {h: 1.0 for h in HAZARDS})

    # Display header
    horizon_label = forecast_choice.replace(" days", "-Day")
    st.markdown(f"### {selected_nbr.title()} — {horizon_label} Forecast")
    st.caption(f"Based on {len(forecast_data):,} historical days matching the {forecast_choice} window")

    # Risk cards (seismic excluded from display)
    qp_display = [h for h in HAZARDS if h != "seismic"]
    hazard_cols = st.columns(len(qp_display))

    for col, h in zip(hazard_cols, qp_display):
        with col:
            base_risk = forecast_data[f"{h}_risk_index"].mean()
            adjustment_factor = nbr_adjustments.get(h, 1.0)
            risk = np.clip(base_risk * adjustment_factor, 0, 100)
            level, color = risk_level(risk)
            st.markdown(f"""
            <div style="background: {COLORS['card_bg']}; padding: 16px; border-radius: 6px;
                        border-left: 4px solid {COLORS[h]}; text-align: center;">
                <div style="color: {COLORS['text_tertiary']}; font-size: 11px; font-weight: 600;
                            text-transform: uppercase;">
                    {HAZARD_LABELS[h]}
                </div>
                <div style="color: {color}; font-size: 24px; font-weight: 600; margin: 8px 0;">
                    {risk:.0f}
                </div>
                <div style="color: {color}; font-size: 12px;">
                    {level}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Neighborhood context map ──────────────────────────────────────────
    # Choropleth highlighting the selected neighborhood with surrounding
    # neighborhoods visible for spatial context.
    st.markdown("### Neighborhood Context")

    # Build GeoDataFrame with incident data for all neighborhoods
    nbr_map_data = []
    for name, p in profiles["neighborhoods"].items():
        flood_exp = p.get("flood_exposure", {})
        fire = p["hazard_call_counts"].get("fire", 0)
        wildfire = p["hazard_call_counts"].get("wildfire", 0)
        nbr_map_data.append({
            "neighborhood": name,
            "total_incidents": p["total_hazard_calls"],
            "fire_incidents": fire + wildfire,
            "area_acres": p["area_acres"],
            "num_basins": flood_exp.get("num_basins", 0),
        })
    map_df = pd.DataFrame(nbr_map_data)
    map_df["incident_rate"] = (map_df["total_incidents"] / map_df["area_acres"].replace(0, 1) * 100).round(1)
    map_df["is_selected"] = (map_df["neighborhood"] == selected_nbr).astype(int)

    gdf = neighborhoods.merge(map_df, on="neighborhood", how="left")

    # Separate selected vs surrounding for distinct styling
    sel_gdf = gdf[gdf["neighborhood"] == selected_nbr]
    other_gdf = gdf[gdf["neighborhood"] != selected_nbr]

    fig_map = go.Figure()

    # Surrounding neighborhoods — muted, with hover info
    if len(other_gdf) > 0:
        hover_other = []
        for _, row in other_gdf.iterrows():
            hover_other.append(
                f"<b>{row['neighborhood'].title()}</b><br>"
                f"EM Incidents: {row['total_incidents']}<br>"
                f"Rate: {row['incident_rate']:.1f}/100ac<br>"
                f"Flood Basins: {row['num_basins']}"
            )
        fig_map.add_trace(go.Choroplethmapbox(
            geojson=other_gdf.geometry.__geo_interface__,
            locations=other_gdf.index,
            z=[0.3] * len(other_gdf),
            colorscale=[[0, "rgba(100,100,120,0.25)"], [1, "rgba(100,100,120,0.25)"]],
            text=hover_other,
            hoverinfo="text",
            marker_opacity=0.4,
            marker_line_width=1,
            marker_line_color=COLORS["border"],
            showscale=False,
        ))

    # Selected neighborhood — highlighted
    if len(sel_gdf) > 0:
        sel_row = sel_gdf.iloc[0]
        hover_sel = (
            f"<b>{sel_row['neighborhood'].title()}</b><br>"
            f"EM Incidents: {sel_row['total_incidents']}<br>"
            f"Rate: {sel_row['incident_rate']:.1f}/100ac<br>"
            f"Flood Basins: {sel_row['num_basins']}"
        )
        fig_map.add_trace(go.Choroplethmapbox(
            geojson=sel_gdf.geometry.__geo_interface__,
            locations=sel_gdf.index,
            z=[1],
            colorscale=[[0, COLORS["primary"]], [1, COLORS["primary"]]],
            text=[hover_sel],
            hoverinfo="text",
            marker_opacity=0.7,
            marker_line_width=2.5,
            marker_line_color="#ffffff",
            showscale=False,
        ))

    # Center on selected neighborhood's centroid
    sel_centroid = sel_gdf.geometry.centroid.iloc[0] if len(sel_gdf) > 0 else None
    center_lat = sel_centroid.y if sel_centroid else 47.979
    center_lon = sel_centroid.x if sel_centroid else -122.202

    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=12,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=400,
        **get_plotly_dark(),
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ── Probabilistic Risk Analysis (disabled) ─────────────────────────────
# Removed from UI: Gamma-Poisson Empirical Bayes rates showed uniform
# 0.02/yr across all neighborhoods because the 911 dispatch data has too
# few hazard-specific events per neighborhood to differentiate. The model-
# based predictions (Quick Predict, Historical Trends) provide better
# risk differentiation. Function preserved for potential future use with
# richer event data.
#
# def page_probabilistic_analysis(predictions, profiles):
#     """Probabilistic Analysis: Bayesian risk assessment with credible intervals."""
#     st.markdown("## Probabilistic Risk Analysis")
#     nbr_stats = compute_neighborhood_stats(predictions, profiles)
#     stats_rows = []
#     for nbr, stats in nbr_stats.items():
#         stats_rows.append({
#             "Neighborhood": nbr,
#             "Fire Rate (per year)": f"{stats['fire_rate']:.2f}",
#             "Fire 90% CI": f"[{stats.get('fire_lower_ci', 0):.2f}, {stats.get('fire_upper_ci', 0):.2f}]",
#             "Flood Rate (per year)": f"{stats['flood_rate']:.2f}",
#             "Flood 90% CI": f"[{stats.get('flood_lower_ci', 0):.2f}, {stats.get('flood_upper_ci', 0):.2f}]",
#         })
#     stats_df = pd.DataFrame(stats_rows)
#     st.dataframe(stats_df, use_container_width=True, hide_index=True)


def page_model_diagnostics():
    """Model Information: Round 3 model details."""
    st.markdown("## Model Information")

    st.markdown("""
    ### AHI Round 3 — PNW Regional Model

    The Everett dashboard uses the PNW (Pacific Northwest) regional model from AHI Round 3,
    a national-scale multi-hazard prediction system trained on 10 million county-day observations
    across 3,109 CONUS counties from 2000 to 2025.

    **Architecture:** Stacked mesh transformer with heat kernel attention
    - **Temporal Module**: Heat kernel diffusion attention
    - **Spatial Module**: County adjacency-masked attention
    - **Gated Coupling**: Learned blend of temporal and spatial signal
    - **Seasonal Bias**: Trainable 5x12 matrix for per-hazard seasonality
    - **Parameters**: 1.3M (under 1.5M target)

    **Performance (PNW Region):**

    | Hazard | AUC |
    |--------|-----|
    | Wildfire | 0.794 |
    | Flood | 0.701 |
    | Severe Wind | 0.711 |
    | Winter Storm | 0.911 |

    **Feature Vector (50 dimensions):**
    - GridMET weather: temperature, precipitation, wind, humidity, ERC, VPD (14 features)
    - Geographic: elevation, forest fraction, urban fraction, population density (4 features)
    - NFHL flood zones: SFHA fraction, V-zone, X-zone, area (5 features)
    - WUI: wildland-urban interface fraction, vegetation cover (6 features)
    - Temporal: day of year, month, year, 3-day rolling means (7 features)
    - Reserved for live data pipeline: AR, snowpack, drought, streamflow (14 features)

    **Calibration Pipeline:**
    1. Temperature scaling (per-state, per-hazard)
    2. County-level seasonal bias (Snohomish County)
    3. Seasonal ceiling (month-specific probability caps)

    **City-Level Adaptation:**
    The model is trained on county-level data. For Everett, we replace county-centroid weather
    with data from Everett's actual GridMET cell (47.98N, 122.20W) while retaining the
    Snohomish County geographic embedding and calibration.
    """)

    st.markdown("### Data Sources")
    st.markdown("""
    | Source | Data | Usage |
    |--------|------|-------|
    | **NOAA** | Storm Events database | Flood, wind, winter labels |
    | **WFIGS** | Wildfire locations | Fire labels |
    | **GridMET** | Daily 4km weather (8 variables) | Temperature, precipitation, wind, humidity |
    | **FEMA NFHL** | National Flood Hazard Layer | Flood zone exposure features |
    | **SILVIS Lab** | Wildland-Urban Interface | WUI exposure features |
    | **Everett EMD** | Fire 911 dispatches | Neighborhood calibration |
    | **Everett Open Data** | Neighborhoods, basins, levees | Spatial context |
    """)


def page_about():
    """About page with data source attribution."""
    st.markdown(f"""
### Model

**AHI Round 3** — PNW Regional Model

| Metric | Value |
|--------|-------|
| Architecture | Stacked mesh transformer with heat kernel attention |
| Parameters | 1,294,547 |
| Training data | 10M county-day observations, 3,109 counties |
| Feature vector | 50 dimensions |
| PNW Fire AUC | 0.794 |
| PNW Flood AUC | 0.701 |
| PNW Wind AUC | 0.711 |
| PNW Winter AUC | 0.911 |

The model predicts daily probability of four hazard types based on weather conditions,
geographic features, and learned seasonal patterns. For this city-level prototype,
county-level weather inputs are replaced with data from Everett's actual
[GridMET](https://www.climatologylab.org/gridmet.html) grid cell (47.98N, 122.20W)
rather than the Snohomish County centroid near Monroe.

### Risk Index

The Risk Index (0-100) represents **relative risk** compared to Everett's own
25-year historical range. An index of 75 means conditions are in the top 25% of
historical risk days for that hazard -- not a 75% probability of an event occurring.

| Level | Index Range | Color |
|-------|-------------|-------|
| Low | 0-24 | Green |
| Moderate | 25-49 | Blue |
| Elevated | 50-74 | Orange |
| High | 75-100 | Red |

Operational trigger language is proposed and requires validation by the OEM director
before operational use.

---

### Data Sources

**Weather & Hazard Model**
- [GridMET](https://www.climatologylab.org/gridmet.html) -- Daily 4km meteorological
  data (2000-present). Variables: temperature, precipitation, wind speed, humidity,
  vapor pressure deficit, energy release component.
- [NOAA Storm Events](https://www.ncdc.noaa.gov/stormevents/) -- Historical hazard event records
- [NWS Forecast API](https://www.weather.gov/documentation/services-web-api) -- 2.5km grid forecasts
- [FEMA NFHL](https://www.fema.gov/flood-maps/national-flood-hazard-layer) -- Flood zone boundaries
- [SILVIS WUI](https://silvis.forest.wisc.edu/data/wui-change/) -- Wildland-urban interface

**City of Everett Open Data** -- [data.everettwa.gov](https://data.everettwa.gov)
- Neighborhood boundaries (19 neighborhoods)
- Neighborhood-Census Tract Bridge (96 mappings)
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

- **Prototype status** -- Uses an existing county-level model with swapped city-level
  weather inputs. A model retrained on city-specific labels would improve accuracy.
- **Earthquake** -- Not displayed. Seismic risk in the PNW is constant background
  probability without seasonal variation, making it not actionable without real-time
  ShakeAlert integration.
- **Landslide** -- Ranked #4 in Everett's CPRI but not in the current AHI feature set.
- **Forecast horizon** -- 30-day forecasts use monthly seasonal averages rather
  than day-specific weather predictions. 14-day horizons are more reliable.
- **Zero-filled features** -- 14 of 50 input features (atmospheric rivers, snowpack,
  drought indices, streamflow) are zero-filled pending live data pipeline integration.

---

### Contact

Developed by Joshua D. Curry — Pierce College BAS Emergency Management capstone project.
    """)


# =====================================================================
# MAIN
# =====================================================================

def main():
    st.set_page_config(
        page_title=PLATFORM_NAME,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()
    profiles, predictions, neighborhoods = load_data()

    # ── Sidebar ──────────────────────────────────────────────────────
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

        # Focus hazard (used by overview + trends)
        st.markdown(f"<p style='color: {COLORS['text_tertiary']}; font-size: 14px; font-weight: 600; "
                    f"text-transform: uppercase; margin-top: 20px; letter-spacing: 0.5px;'>"
                    f"Focus Hazard</p>", unsafe_allow_html=True)

        selected_hazard = st.selectbox(
            "Hazard",
            [h for h in HAZARDS if h != "seismic"],
            format_func=lambda h: HAZARD_LABELS[h],
            label_visibility="collapsed",
        )

        # Navigation: Analysis
        st.markdown(f"<p style='color: {COLORS['text_tertiary']}; font-size: 14px; font-weight: 600; "
                    f"text-transform: uppercase; margin-top: 20px; letter-spacing: 0.5px;'>"
                    f"Analysis</p>", unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state.page = "overview"

        if st.button("Risk Overview", use_container_width=True):
            st.session_state.page = "overview"
        if st.button("Neighborhood Map", use_container_width=True):
            st.session_state.page = "map"
        if st.button("Historical Trends", use_container_width=True):
            st.session_state.page = "trends"

        # Navigation: AI Tools
        st.markdown(f"<p style='color: {COLORS['text_tertiary']}; font-size: 14px; font-weight: 600; "
                    f"text-transform: uppercase; margin-top: 20px; letter-spacing: 0.5px;'>"
                    f"AI Tools</p>", unsafe_allow_html=True)

        if st.button("Quick Predict", use_container_width=True):
            st.session_state.page = "predict"
        # Probabilistic Analysis removed — Gamma-Poisson rates were not
        # model-derived and showed uninformative 0.02/yr for all neighborhoods.
        # if st.button("Probabilistic Analysis", use_container_width=True):
        #     st.session_state.page = "bayes"
        if st.button("Model Info", use_container_width=True):
            st.session_state.page = "diagnostics"
        if st.button("About & Data Sources", use_container_width=True):
            st.session_state.page = "about"

    # ── Header ──────────────────────────────────────────────────────
    render_header()

    # ── Page routing ────────────────────────────────────────────────
    page = st.session_state.get("page", "overview")

    if page == "overview":
        page_risk_overview(predictions, profiles, selected_hazard)
    elif page == "map":
        page_neighborhood_map(profiles, neighborhoods)
    elif page == "trends":
        page_historical_trends(predictions, selected_hazard)
    elif page == "predict":
        page_quick_predict(predictions, profiles, neighborhoods, selected_hazard)
    # elif page == "bayes":
    #     page_probabilistic_analysis(predictions, profiles)
    elif page == "diagnostics":
        page_model_diagnostics()
    elif page == "about":
        page_about()


if __name__ == "__main__":
    main()
