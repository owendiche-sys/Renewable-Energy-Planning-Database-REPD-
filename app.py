import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

# =========================
# Page config (light theme)
# =========================
st.set_page_config(page_title="Wind Turbine SCADA Dashboard", layout="wide")

BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "rgba(17,24,39,0.65)"
BORDER = "rgba(15,23,42,0.08)"

st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
}}
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 1.5rem;
}}
#MainMenu {{visibility:hidden;}}
footer {{visibility:hidden;}}

section[data-testid="stSidebar"] > div {{
    border-right: 1px solid {BORDER};
}}

.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 16px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}}
.small {{
    color: {MUTED};
    font-size: 12px;
}}
</style>
""",
    unsafe_allow_html=True,
)

def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
<div class="card">
  <div style="color:{MUTED}; font-weight:600; font-size:14px;">{title}</div>
  <div style="color:{TEXT}; font-weight:800; font-size:28px; margin-top:6px;">{value}</div>
  <div style="color:{MUTED}; font-size:12px; margin-top:6px;">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================
# Data loading (single folder)
# =========================
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "wind_turbine_scada.csv"

@st.cache_data(show_spinner=False)
def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(uploaded_file)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Map columns (based on your notebook logic)
    column_map = {}
    for col in df.columns:
        low = col.lower()
        if "date" in low and "time" in low:
            column_map[col] = "datetime"
        elif "activepower" in low or "active power" in low:
            column_map[col] = "active_power_kw"
        elif "wind speed" in low:
            column_map[col] = "wind_speed_ms"
        elif "theoretical_power" in low or "theoretical power" in low:
            column_map[col] = "theoretical_power_kw"
        elif "wind direction" in low:
            column_map[col] = "wind_direction_deg"

    df = df.rename(columns=column_map)

    # Parse datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Convert numeric columns
    for col in ["active_power_kw", "wind_speed_ms", "theoretical_power_kw", "wind_direction_deg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing key fields
    if "datetime" in df.columns:
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return df

def compute_underperformance(df: pd.DataFrame, quantile_threshold: float = 0.90) -> pd.DataFrame:
    df = df.copy()
    if {"active_power_kw", "theoretical_power_kw"}.issubset(df.columns):
        df["power_gap_kw"] = df["theoretical_power_kw"] - df["active_power_kw"]
        thr = df["power_gap_kw"].quantile(quantile_threshold)
        df["underperforming"] = df["power_gap_kw"] > thr
    else:
        df["power_gap_kw"] = np.nan
        df["underperforming"] = False
    return df

def fmt_num(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) >= 1e6:
        return f"{x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.2f}"

# =========================
# Header
# =========================
st.markdown("## Wind Turbine SCADA Performance Dashboard")
st.caption(
    "An interactive dashboard for exploring wind turbine output, power curves, time trends, and underperformance behaviour using SCADA data."
)
st.write("")

# =========================
# Sidebar
# =========================
st.sidebar.title("Controls")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard Summary", "Power Curve", "Time Trends", "Underperformance", "Data"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Data source")

use_default = st.sidebar.checkbox("Load wind_turbine_scada.csv on startup", value=True)

if use_default:
    if not DEFAULT_DATA.exists():
        st.error("wind_turbine_scada.csv not found in the same folder as app.py. Place it next to app.py and rerun.")
        st.stop()
    raw = load_csv_with_fallback(DEFAULT_DATA)
    source_label = "Loaded: wind_turbine_scada.csv"
else:
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.info("Upload a CSV or enable the default dataset option.")
        st.stop()
    raw = load_uploaded_csv(up)
    source_label = "Loaded: upload"

df = standardize_columns(raw)
df = compute_underperformance(df, quantile_threshold=0.90)

st.sidebar.caption(source_label)

# Required columns check (soft)
required = ["datetime", "active_power_kw", "wind_speed_ms"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.warning(
        "Some expected columns were not detected after standardisation. "
        f"Missing: {', '.join(missing)}. The dashboard will still run, but some charts may be limited."
    )

# Filters
st.sidebar.divider()
st.sidebar.subheader("Filters")

if "datetime" in df.columns and df["datetime"].notna().any():
    min_dt = df["datetime"].min()
    max_dt = df["datetime"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    start_date, end_date = None, None

if "wind_speed_ms" in df.columns and df["wind_speed_ms"].notna().any():
    ws_min = float(np.nanquantile(df["wind_speed_ms"], 0.01))
    ws_max = float(np.nanquantile(df["wind_speed_ms"], 0.99))
    wind_speed_range = st.sidebar.slider("Wind speed range (m/s)", ws_min, ws_max, (ws_min, ws_max))
else:
    wind_speed_range = None

if "wind_direction_deg" in df.columns and df["wind_direction_deg"].notna().any():
    wd_min = float(np.nanquantile(df["wind_direction_deg"], 0.01))
    wd_max = float(np.nanquantile(df["wind_direction_deg"], 0.99))
    wind_dir_range = st.sidebar.slider("Wind direction range (deg)", wd_min, wd_max, (wd_min, wd_max))
else:
    wind_dir_range = None

only_under = st.sidebar.checkbox("Show underperforming records only", value=False)
scatter_n = st.sidebar.slider("Scatter sample size", 1000, 20000, 5000, step=1000)

# Apply filters
dff = df.copy()

if start_date is not None and "datetime" in dff.columns:
    dff = dff[(dff["datetime"] >= start_date) & (dff["datetime"] <= end_date)]

if wind_speed_range is not None and "wind_speed_ms" in dff.columns:
    dff = dff[(dff["wind_speed_ms"] >= wind_speed_range[0]) & (dff["wind_speed_ms"] <= wind_speed_range[1])]

if wind_dir_range is not None and "wind_direction_deg" in dff.columns:
    dff = dff[(dff["wind_direction_deg"] >= wind_dir_range[0]) & (dff["wind_direction_deg"] <= wind_dir_range[1])]

if only_under and "underperforming" in dff.columns:
    dff = dff[dff["underperforming"] == True]

# =========================
# KPI row
# =========================
n_rows = len(dff)
date_span = "—"
if "datetime" in dff.columns and n_rows > 0:
    date_span = f"{dff['datetime'].min().date()} to {dff['datetime'].max().date()}"

avg_power = float(dff["active_power_kw"].mean()) if "active_power_kw" in dff.columns and n_rows > 0 else np.nan
max_power = float(dff["active_power_kw"].max()) if "active_power_kw" in dff.columns and n_rows > 0 else np.nan
avg_ws = float(dff["wind_speed_ms"].mean()) if "wind_speed_ms" in dff.columns and n_rows > 0 else np.nan

under_rate = np.nan
if "underperforming" in dff.columns and n_rows > 0:
    under_rate = float(dff["underperforming"].mean() * 100)

k1, k2, k3, k4 = st.columns(4, gap="large")
with k1:
    kpi_card("Records in view", f"{n_rows:,}", f"Date range: {date_span}")
with k2:
    kpi_card("Average active power (kW)", fmt_num(avg_power), "Mean power output for selected records")
with k3:
    kpi_card("Max active power (kW)", fmt_num(max_power), "Maximum observed power output in view")
with k4:
    kpi_card("Underperformance rate", f"{under_rate:.2f}%" if np.isfinite(under_rate) else "—", "Share flagged as high power gap")

st.write("")

# =========================
# Pages
# =========================
if page == "Dashboard Summary":
    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Project Summary")
        st.write(
            """
This dashboard provides an overview of turbine behaviour using SCADA measurements:
- Output performance and environmental conditions (wind speed and direction)
- Power curve behaviour (actual vs theoretical when available)
- Time-based trends using daily and monthly aggregation
- Underperformance detection using a high power-gap threshold (top decile by default)
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data Overview")
        st.dataframe(dff.head(30), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Power Curve":
    st.markdown("### Power Curve Analysis")

    if not {"wind_speed_ms", "active_power_kw"}.issubset(dff.columns) or len(dff) == 0:
        st.info("Power curve charts require wind_speed_ms and active_power_kw with at least one record in view.")
    else:
        plot_df = dff.sample(min(scatter_n, len(dff)), random_state=42)

        c1, c2 = st.columns([1.35, 1.0], gap="large")

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Actual power vs wind speed")
            color_col = "underperforming" if "underperforming" in plot_df.columns else None
            fig = px.scatter(
                plot_df,
                x="wind_speed_ms",
                y="active_power_kw",
                color=color_col,
                opacity=0.35,
                labels={"wind_speed_ms": "Wind speed (m/s)", "active_power_kw": "Active power (kW)"},
            )
            fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="Underperforming")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Binned curve summary")
            st.caption("Mean power within wind-speed bins (reduces noise and is easy to interpret).")

            # Binning
            bins = st.slider("Number of wind-speed bins", 10, 60, 30)
            ws = plot_df["wind_speed_ms"].astype(float)
            pw = plot_df["active_power_kw"].astype(float)

            binned = pd.DataFrame({"wind_speed_ms": ws, "active_power_kw": pw}).dropna()
            binned["bin"] = pd.cut(binned["wind_speed_ms"], bins=bins)
            curve = binned.groupby("bin", observed=True).agg(
                wind_speed_mean=("wind_speed_ms", "mean"),
                power_mean=("active_power_kw", "mean"),
                n=("active_power_kw", "size"),
            ).reset_index(drop=True)

            fig2 = px.line(
                curve,
                x="wind_speed_mean",
                y="power_mean",
                markers=True,
                labels={"wind_speed_mean": "Wind speed (m/s)", "power_mean": "Mean active power (kW)"},
            )
            fig2.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("<div class='small'>Higher bins with low sample size can be less stable.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        if "theoretical_power_kw" in dff.columns:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Actual vs theoretical power (sampled)")
            plot_df2 = dff.sample(min(scatter_n, len(dff)), random_state=7)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=plot_df2["wind_speed_ms"], y=plot_df2["theoretical_power_kw"],
                mode="markers", name="Theoretical", opacity=0.30
            ))
            fig3.add_trace(go.Scatter(
                x=plot_df2["wind_speed_ms"], y=plot_df2["active_power_kw"],
                mode="markers", name="Actual", opacity=0.30
            ))
            fig3.update_layout(
                height=520,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Wind speed (m/s)",
                yaxis_title="Power (kW)",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Time Trends":
    st.markdown("### Time-Based Trends")

    if "datetime" not in dff.columns or "active_power_kw" not in dff.columns or len(dff) == 0:
        st.info("Time trend charts require datetime and active_power_kw with at least one record in view.")
    else:
        tmp = dff[["datetime", "active_power_kw"]].dropna().copy()
        tmp = tmp.sort_values("datetime").set_index("datetime")

        daily = tmp["active_power_kw"].resample("D").mean().dropna()
        monthly = tmp["active_power_kw"].resample("MS").mean().dropna()

        a, b = st.columns([1.35, 1.0], gap="large")

        with a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Daily mean active power")
            fig = px.line(
                daily.reset_index(),
                x="datetime",
                y="active_power_kw",
                labels={"datetime": "Date", "active_power_kw": "Daily mean active power (kW)"},
            )
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Monthly mean active power")
            fig2 = px.bar(
                monthly.reset_index(),
                x="datetime",
                y="active_power_kw",
                labels={"datetime": "Month", "active_power_kw": "Monthly mean active power (kW)"},
            )
            fig2.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Context view (first N records)")
        st.caption("A high-level time series segment for quick inspection.")
        n = st.slider("Number of records to display", 500, 20000, 5000, step=500)
        seg = dff.sort_values("datetime").head(min(n, len(dff)))
        fig3 = px.line(
            seg,
            x="datetime",
            y="active_power_kw",
            labels={"datetime": "Time", "active_power_kw": "Active power (kW)"},
        )
        fig3.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Underperformance":
    st.markdown("### Underperformance Analysis")

    if not {"active_power_kw", "theoretical_power_kw"}.issubset(dff.columns) or len(dff) == 0:
        st.info("Underperformance analysis requires active_power_kw and theoretical_power_kw.")
    else:
        a, b = st.columns([1.1, 1.0], gap="large")

        with a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Power gap distribution")
            g = dff["power_gap_kw"].dropna()
            fig = px.histogram(
                g,
                nbins=60,
                labels={"value": "Power gap (kW)"},
            )
            fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Top underperforming intervals")
            show_n = st.slider("Rows to show", 5, 50, 15)
            cols = ["datetime", "wind_speed_ms", "active_power_kw", "theoretical_power_kw", "power_gap_kw"]
            existing = [c for c in cols if c in dff.columns]
            top = dff.sort_values("power_gap_kw", ascending=False).head(show_n)[existing]
            st.dataframe(top, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Where underperformance happens (wind speed vs actual power)")
        plot_df = dff.sample(min(scatter_n, len(dff)), random_state=123)

        fig2 = px.scatter(
            plot_df,
            x="wind_speed_ms",
            y="active_power_kw",
            color="underperforming",
            opacity=0.40,
            labels={"wind_speed_ms": "Wind speed (m/s)", "active_power_kw": "Active power (kW)"},
        )
        fig2.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="Underperforming")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            "<div class='small'>Underperforming records are defined as the top decile of power-gap values within the current view.</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:  # Data page
    st.markdown("### Data")
    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Column detection")
        st.write("The dashboard standardises common SCADA column names into:")
        st.write("- datetime")
        st.write("- active_power_kw")
        st.write("- wind_speed_ms")
        st.write("- theoretical_power_kw (optional)")
        st.write("- wind_direction_deg (optional)")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Columns found")
        st.dataframe(pd.DataFrame({"column": df.columns}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Full preview")
    st.dataframe(dff.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
