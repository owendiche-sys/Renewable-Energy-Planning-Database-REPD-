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
    padding-top: 3.25rem;  /* moves header down so it never clips */
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


def fmt_num(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) >= 1e6:
        return f"{x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.2f}"


def pct(x: float) -> str:
    return "—" if (x is None or not np.isfinite(x)) else f"{x:.2f}%"


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
    """
    Works with your dataset columns:
      - Date/Time
      - LV ActivePower (kW)
      - Wind Speed (m/s)
      - Theoretical_Power_Curve (KWh)
      - Wind Direction (°)
    And still supports other common SCADA naming variants.
    """
    df = df.copy()

    column_map = {}
    for col in df.columns:
        low = col.lower().strip()

        if low in {"date/time", "datetime", "date time", "timestamp"}:
            column_map[col] = "datetime"
        elif "date" in low and "time" in low:
            column_map[col] = "datetime"

        elif "activepower" in low or "active power" in low:
            column_map[col] = "active_power_kw"
        elif "lv activepower" in low:
            column_map[col] = "active_power_kw"

        elif "wind speed" in low:
            column_map[col] = "wind_speed_ms"

        elif "theoretical_power_curve" in low or "theoretical power" in low:
            column_map[col] = "theoretical_power_ref"
        elif "theoretical" in low and "power" in low:
            column_map[col] = "theoretical_power_ref"

        elif "wind direction" in low or "direction" in low:
            column_map[col] = "wind_direction_deg"

    df = df.rename(columns=column_map)

    # Parse datetime robustly
    if "datetime" in df.columns:
        s = df["datetime"].astype(str)
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        # fallback for "01 01 2018 00:00" style
        if dt.isna().mean() > 0.2:
            dt2 = pd.to_datetime(s, errors="coerce", format="%d %m %Y %H:%M")
            dt = dt.fillna(dt2)
        df["datetime"] = dt

    # Convert numeric columns
    for col in ["active_power_kw", "wind_speed_ms", "theoretical_power_ref", "wind_direction_deg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing essential fields
    if "datetime" in df.columns:
        df = df.dropna(subset=["datetime"]).sort_values("datetime")

    return df


def add_power_gap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"active_power_kw", "theoretical_power_ref"}.issubset(df.columns):
        df["power_gap"] = df["theoretical_power_ref"] - df["active_power_kw"]
    else:
        df["power_gap"] = np.nan
    return df


def circular_mode_deg(series: pd.Series, bin_size: int = 10) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "—"
    b = (np.floor(s / bin_size) * bin_size).astype(int)
    mode_bin = int(b.value_counts().idxmax())
    return f"{mode_bin}–{mode_bin + bin_size}°"


# =========================
# Header
# =========================
st.markdown("## Wind Turbine SCADA Performance Dashboard")
st.caption(
    "An interactive dashboard for exploring turbine output, power curve behaviour, time trends, and underperformance patterns using SCADA data."
)
st.write("")


# =========================
# Sidebar
# =========================
st.sidebar.title("Controls")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard Summary", "Insights", "Power Curve", "Time Trends", "Underperformance", "Data"],
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
df = add_power_gap(df)

st.sidebar.caption(source_label)

# Soft required columns check
required = ["datetime", "active_power_kw", "wind_speed_ms"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.warning(
        "Some expected columns were not detected after standardisation. "
        f"Missing: {', '.join(missing)}. Some sections may be limited."
    )

st.sidebar.divider()
st.sidebar.subheader("Filters")

# Date filter
if "datetime" in df.columns and df["datetime"].notna().any():
    min_dt = df["datetime"].min()
    max_dt = df["datetime"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    start_date, end_date = None, None

# Wind speed filter
if "wind_speed_ms" in df.columns and df["wind_speed_ms"].notna().any():
    ws_min = float(np.nanquantile(df["wind_speed_ms"], 0.01))
    ws_max = float(np.nanquantile(df["wind_speed_ms"], 0.99))
    wind_speed_range = st.sidebar.slider("Wind speed range (m/s)", ws_min, ws_max, (ws_min, ws_max))
else:
    wind_speed_range = None

# Wind direction filter
if "wind_direction_deg" in df.columns and df["wind_direction_deg"].notna().any():
    wd_min = float(np.nanquantile(df["wind_direction_deg"], 0.01))
    wd_max = float(np.nanquantile(df["wind_direction_deg"], 0.99))
    wind_dir_range = st.sidebar.slider("Wind direction range (deg)", wd_min, wd_max, (wd_min, wd_max))
else:
    wind_dir_range = None

# Underperformance controls
st.sidebar.divider()
st.sidebar.subheader("Underperformance logic")
quantile_threshold = st.sidebar.slider("Flag top power-gap quantile", 0.70, 0.99, 0.90, step=0.01)
only_under = st.sidebar.checkbox("Show underperforming records only", value=False)

# Plot controls
st.sidebar.divider()
st.sidebar.subheader("Plot settings")
scatter_n = st.sidebar.slider("Scatter sample size", 1000, 20000, 5000, step=1000)

# Apply filters (base view first)
base = df.copy()

if start_date is not None and "datetime" in base.columns:
    base = base[(base["datetime"] >= start_date) & (base["datetime"] <= end_date)]

if wind_speed_range is not None and "wind_speed_ms" in base.columns:
    base = base[(base["wind_speed_ms"] >= wind_speed_range[0]) & (base["wind_speed_ms"] <= wind_speed_range[1])]

if wind_dir_range is not None and "wind_direction_deg" in base.columns:
    base = base[(base["wind_direction_deg"] >= wind_dir_range[0]) & (base["wind_direction_deg"] <= wind_dir_range[1])]

# Compute underperformance within the current view (so the flag matches the filters)
dff = base.copy()
if "power_gap" in dff.columns and dff["power_gap"].notna().any():
    thr = float(dff["power_gap"].quantile(quantile_threshold))
    dff["underperforming"] = dff["power_gap"] > thr
else:
    dff["underperforming"] = False
    thr = np.nan

if only_under:
    dff = dff[dff["underperforming"] == True]

# Safety
if len(dff) == 0:
    st.warning("No records match the selected filters.")
    st.stop()


# =========================
# KPI row (only on Summary + Insights)
# =========================
def render_kpis(frame: pd.DataFrame):
    n_rows = len(frame)
    date_span = "—"
    if "datetime" in frame.columns and frame["datetime"].notna().any():
        date_span = f"{frame['datetime'].min().date()} to {frame['datetime'].max().date()}"

    avg_power = float(frame["active_power_kw"].mean()) if "active_power_kw" in frame.columns else np.nan
    max_power = float(frame["active_power_kw"].max()) if "active_power_kw" in frame.columns else np.nan
    avg_ws = float(frame["wind_speed_ms"].mean()) if "wind_speed_ms" in frame.columns else np.nan

    under_rate = float(frame["underperforming"].mean() * 100) if "underperforming" in frame.columns else np.nan

    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        kpi_card("Records in view", f"{n_rows:,}", f"Date range: {date_span}")
    with k2:
        kpi_card("Average active power (kW)", fmt_num(avg_power), "Mean output for selected records")
    with k3:
        kpi_card("Average wind speed (m/s)", fmt_num(avg_ws), "Mean wind speed for selected records")
    with k4:
        kpi_card("Underperformance rate", pct(under_rate), "Share flagged as high power gap")


# =========================
# Pages
# =========================
if page in {"Dashboard Summary", "Insights"}:
    render_kpis(dff)
    st.write("")

if page == "Dashboard Summary":
    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Project Summary")
        st.write(
            """
This dashboard provides a clear view of turbine performance using SCADA measurements:
- Output behaviour under different wind conditions
- Power curve patterns (actual vs reference curve)
- Time-based trends using daily and monthly aggregation
- Underperformance detection using high power-gap intervals within the current view
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data Overview")
        st.dataframe(dff.head(30), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Insights":
    st.markdown("### Insights")
    st.caption("Data-driven insights first, then analytical insights based on the reference curve and underperformance logic.")
    st.write("")

    # ---------- Data-driven insights ----------
    st.markdown("#### Data-driven insights")
    a, b = st.columns([1.35, 1.0], gap="large")

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Daily output pattern (mean active power)")
        if "datetime" in dff.columns and "active_power_kw" in dff.columns:
            tmp = dff[["datetime", "active_power_kw"]].dropna().sort_values("datetime").set_index("datetime")
            daily = tmp["active_power_kw"].resample("D").mean().dropna()

            fig = px.line(
                daily.reset_index(),
                x="datetime",
                y="active_power_kw",
                labels={"datetime": "Date", "active_power_kw": "Daily mean active power (kW)"},
            )
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            if len(daily) > 0:
                best_day = daily.idxmax().date()
                st.info(f"Highest daily mean output in view: {best_day}")
        else:
            st.info("This chart requires datetime and active_power_kw.")
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Quick takeaways")
        missing_pct = float(dff.isna().sum().sum() / (dff.shape[0] * dff.shape[1]) * 100)

        ws_corr = np.nan
        if {"wind_speed_ms", "active_power_kw"}.issubset(dff.columns):
            tmp2 = dff[["wind_speed_ms", "active_power_kw"]].dropna()
            if len(tmp2) > 10:
                ws_corr = float(tmp2["wind_speed_ms"].corr(tmp2["active_power_kw"]))

        dom_dir = "—"
        if "wind_direction_deg" in dff.columns:
            dom_dir = circular_mode_deg(dff["wind_direction_deg"], bin_size=10)

        st.write(
            f"""
- Missing data in view: **{missing_pct:.2f}%**
- Wind speed vs active power correlation: **{ws_corr:.2f}**  
- Dominant wind direction band: **{dom_dir}**
            """.strip()
        )
        st.markdown("<div class='small'>Correlation is a quick signal of how strongly wind speed aligns with output in the selected view.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ---------- Analytical insights (reference curve / underperformance) ----------
    st.markdown("#### Analytical insights")
    c1, c2 = st.columns([1.2, 1.0], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Where the biggest gaps happen (by wind-speed bin)")
        if {"wind_speed_ms", "power_gap"}.issubset(dff.columns) and dff["power_gap"].notna().any():
            tmp = dff[["wind_speed_ms", "power_gap", "underperforming"]].dropna()
            # 0.5 m/s bins (readable)
            tmp["ws_bin"] = pd.cut(tmp["wind_speed_ms"], bins=np.arange(tmp["wind_speed_ms"].min(), tmp["wind_speed_ms"].max() + 0.5, 0.5))
            bybin = tmp.groupby("ws_bin", observed=True).agg(
                wind_speed_mean=("wind_speed_ms", "mean"),
                gap_mean=("power_gap", "mean"),
                under_rate=("underperforming", "mean"),
                n=("power_gap", "size"),
            ).reset_index(drop=True)

            # Keep bins with reasonable sample size
            bybin = bybin[bybin["n"] >= max(30, int(0.002 * len(tmp)))].copy()
            if len(bybin) == 0:
                st.info("Not enough data per bin after filtering. Increase date range or loosen filters.")
            else:
                fig = px.line(
                    bybin.sort_values("wind_speed_mean"),
                    x="wind_speed_mean",
                    y="gap_mean",
                    markers=True,
                    labels={"wind_speed_mean": "Wind speed (m/s)", "gap_mean": "Mean power gap (reference - actual)"},
                )
                fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

                worst = bybin.sort_values("gap_mean", ascending=False).head(1)
                if len(worst) == 1:
                    ws_worst = float(worst["wind_speed_mean"].iloc[0])
                    gap_worst = float(worst["gap_mean"].iloc[0])
                    st.info(f"Largest average gap occurs around **{ws_worst:.2f} m/s** (mean gap ≈ **{gap_worst:.2f}**).")
        else:
            st.info("Analytical gap insights require theoretical reference values and active power.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top underperforming intervals")
        if {"datetime", "wind_speed_ms", "active_power_kw", "theoretical_power_ref", "power_gap"}.issubset(dff.columns) and dff["power_gap"].notna().any():
            show_n = st.slider("Rows to show", 5, 30, 10)
            cols = ["datetime", "wind_speed_ms", "active_power_kw", "theoretical_power_ref", "power_gap"]
            top = dff.sort_values("power_gap", ascending=False).head(show_n)[cols]
            top = top.rename(columns={"theoretical_power_ref": "theoretical_ref"})
            st.dataframe(top, use_container_width=True, hide_index=True)
            st.markdown(
                f"<div class='small'>Underperformance flag uses the top {quantile_threshold:.0%} of power-gap values within the current filtered view.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("This table requires datetime, wind speed, active power, and reference curve values.")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Power Curve":
    st.markdown("### Power Curve Analysis")

    if not {"wind_speed_ms", "active_power_kw"}.issubset(dff.columns):
        st.info("Power curve charts require wind_speed_ms and active_power_kw.")
        st.stop()

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

        bins = st.slider("Number of wind-speed bins", 10, 60, 30)
        ws = pd.to_numeric(plot_df["wind_speed_ms"], errors="coerce")
        pw = pd.to_numeric(plot_df["active_power_kw"], errors="coerce")

        binned = pd.DataFrame({"wind_speed_ms": ws, "active_power_kw": pw}).dropna()
        binned["bin"] = pd.cut(binned["wind_speed_ms"], bins=bins)

        curve = (
            binned.groupby("bin", observed=True)
            .agg(
                wind_speed_mean=("wind_speed_ms", "mean"),
                power_mean=("active_power_kw", "mean"),
                n=("active_power_kw", "size"),
            )
            .reset_index(drop=True)
        )

        fig2 = px.line(
            curve,
            x="wind_speed_mean",
            y="power_mean",
            markers=True,
            labels={"wind_speed_mean": "Wind speed (m/s)", "power_mean": "Mean active power (kW)"},
        )
        fig2.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<div class='small'>Bins with low sample size can be less stable.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    if "theoretical_power_ref" in dff.columns:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Actual vs reference curve values (sampled)")
        st.caption("This compares measured output against the reference curve values included in the dataset.")

        plot_df2 = dff.sample(min(scatter_n, len(dff)), random_state=7)

        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=plot_df2["wind_speed_ms"],
                y=plot_df2["theoretical_power_ref"],
                mode="markers",
                name="Reference curve",
                opacity=0.30,
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=plot_df2["wind_speed_ms"],
                y=plot_df2["active_power_kw"],
                mode="markers",
                name="Actual",
                opacity=0.30,
            )
        )
        fig3.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Wind speed (m/s)",
            yaxis_title="Value",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Time Trends":
    st.markdown("### Time-Based Trends")

    if "datetime" not in dff.columns or "active_power_kw" not in dff.columns:
        st.info("Time trend charts require datetime and active_power_kw.")
        st.stop()

    tmp = dff[["datetime", "active_power_kw"]].dropna().sort_values("datetime").set_index("datetime")
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

    if not {"active_power_kw", "theoretical_power_ref", "power_gap"}.issubset(dff.columns) or dff["power_gap"].notna().sum() == 0:
        st.info("Underperformance analysis requires active_power_kw and theoretical reference values.")
        st.stop()

    a, b = st.columns([1.1, 1.0], gap="large")

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Power gap distribution (reference - actual)")
        g = dff["power_gap"].dropna()
        fig = px.histogram(g, nbins=60, labels={"value": "Power gap"})
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Current underperformance threshold (quantile): {quantile_threshold:.0%} • gap threshold value: {fmt_num(thr)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top underperforming intervals")
        show_n = st.slider("Rows to show", 5, 50, 15)
        cols = ["datetime", "wind_speed_ms", "active_power_kw", "theoretical_power_ref", "power_gap"]
        existing = [c for c in cols if c in dff.columns]
        top = dff.sort_values("power_gap", ascending=False).head(show_n)[existing].copy()
        top = top.rename(columns={"theoretical_power_ref": "theoretical_ref"})
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
        "<div class='small'>Underperforming records are defined as the highest power-gap values within the current filtered view.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:  # Data
    st.markdown("### Data")
    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Column standardisation")
        st.write("The dashboard standardises common SCADA column names into:")
        st.write("- datetime")
        st.write("- active_power_kw")
        st.write("- wind_speed_ms")
        st.write("- theoretical_power_ref (optional)")
        st.write("- wind_direction_deg (optional)")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Columns found after standardisation")
        st.dataframe(pd.DataFrame({"column": df.columns}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Full preview (first 200 rows)")
    st.dataframe(dff.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
