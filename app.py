from __future__ import annotations

import io
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Wind Turbine SCADA Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "wind_turbine_scada.csv"

BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "rgba(17,24,39,0.68)"
BORDER = "rgba(15,23,42,0.10)"


# =========================================================
# Styling
# =========================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --bg: {BG};
            --panel: {CARD};
            --text: {TEXT};
            --muted: {MUTED};
            --border: {BORDER};
            --shadow: 0 10px 30px rgba(15,23,42,0.06);
            --radius: 18px;
            --accent: #1d4ed8;
            --accent-soft: rgba(29,78,216,0.08);
          }}

          html, body, [data-testid="stAppViewContainer"] {{
            background: var(--bg) !important;
            color: var(--text) !important;
          }}

          [data-testid="stHeader"] {{
            background: rgba(246,248,252,0.82);
          }}

          .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
            max-width: 1400px;
          }}

          #MainMenu {{ visibility: hidden; }}
          footer {{ visibility: hidden; }}

          section[data-testid="stSidebar"] > div {{
            background: #ffffff !important;
            border-right: 1px solid var(--border);
          }}

          .hero {{
            background: linear-gradient(135deg, #ffffff 0%, #f9fbff 100%);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 24px 24px 18px 24px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
          }}

          .hero-title {{
            font-size: 30px;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0 0 8px 0;
            color: var(--text);
          }}

          .hero-sub {{
            margin: 0;
            font-size: 15px;
            line-height: 1.6;
            color: var(--muted);
            max-width: 980px;
          }}

          .hero-strip {{
            margin-top: 14px;
            padding: 10px 12px;
            border-radius: 14px;
            background: var(--accent-soft);
            border: 1px solid rgba(29,78,216,0.12);
            color: #1e3a8a;
            font-size: 13px;
          }}

          .section-title {{
            font-size: 18px;
            font-weight: 800;
            color: var(--text);
            margin: 0 0 10px 0;
          }}

          .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 16px;
          }}

          .card-title {{
            margin: 0 0 6px 0;
            font-size: 16px;
            font-weight: 800;
            color: var(--text);
          }}

          .card-sub {{
            margin: 0 0 12px 0;
            color: var(--muted);
            font-size: 13px;
            line-height: 1.5;
          }}

          .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 10px;
          }}

          @media (max-width: 1280px) {{
            .kpi-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
          }}

          @media (max-width: 700px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
          }}

          @media (max-width: 540px) {{
            .kpi-grid {{ grid-template-columns: repeat(1, minmax(0, 1fr)); }}
          }}

          .kpi-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: var(--shadow);
            padding: 14px;
            min-height: 108px;
          }}

          .kpi-label {{
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 8px;
          }}

          .kpi-value {{
            font-size: 24px;
            font-weight: 800;
            line-height: 1.1;
            color: var(--text);
            margin-bottom: 8px;
          }}

          .kpi-note {{
            font-size: 12px;
            color: var(--muted);
            line-height: 1.45;
          }}

          .insight-list {{
            margin: 0;
            padding-left: 18px;
          }}

          .insight-list li {{
            margin-bottom: 8px;
            line-height: 1.55;
            color: var(--text);
          }}

          .badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 6px;
          }}

          .badge {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: #f8fafc;
            border: 1px solid var(--border);
            color: var(--text);
            font-size: 12px;
          }}

          .small-note {{
            color: var(--muted);
            font-size: 12px;
          }}

          .js-plotly-plot .plotly .modebar {{
            opacity: 0.08;
          }}

          .js-plotly-plot .plotly:hover .modebar {{
            opacity: 1;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =========================================================
# Formatting helpers
# =========================================================
def fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or pd.isna(x) or not np.isfinite(x):
        return "N/A"
    if digits == 0:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


def fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None or pd.isna(x) or not np.isfinite(x):
        return "N/A"
    return f"{x * 100:,.{digits}f}%"


def safe_divide(a: float, b: float) -> float:
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


def esc(text: str) -> str:
    return escape(str(text))


# =========================================================
# UI helpers
# =========================================================
def render_section_title(text: str) -> None:
    st.markdown(f"<div class='section-title'>{esc(text)}</div>", unsafe_allow_html=True)


def render_kpis(items: List[Tuple[str, str, str]]) -> None:
    blocks: List[str] = []
    for label, value, note in items:
        blocks.append(
            "<div class='kpi-card'>"
            f"<div class='kpi-label'>{esc(label)}</div>"
            f"<div class='kpi-value'>{esc(value)}</div>"
            f"<div class='kpi-note'>{esc(note)}</div>"
            "</div>"
        )
    st.markdown("<div class='kpi-grid'>" + "".join(blocks) + "</div>", unsafe_allow_html=True)


def render_list_card(title: str, items: List[str], subtitle: str = "") -> None:
    html = "<div class='card'>"
    html += f"<div class='card-title'>{esc(title)}</div>"
    if subtitle:
        html += f"<div class='card-sub'>{esc(subtitle)}</div>"
    html += "<ul class='insight-list'>"
    for item in items:
        html += f"<li>{esc(item)}</li>"
    html += "</ul></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_badges(items: List[str]) -> None:
    html = "<div class='badge-row'>"
    for item in items:
        html += f"<span class='badge'>{esc(item)}</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# Data loading
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv_with_fallback_from_path(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_with_fallback_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(file_bytes))


# =========================================================
# Column standardisation and feature engineering
# =========================================================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    column_map: Dict[str, str] = {}
    for col in out.columns:
        low = str(col).lower().strip()

        if low in {"date/time", "datetime", "date time", "timestamp"}:
            column_map[col] = "datetime"
        elif "date" in low and "time" in low:
            column_map[col] = "datetime"

        elif "lv activepower" in low or "activepower" in low or "active power" in low:
            column_map[col] = "active_power_kw"

        elif "wind speed" in low:
            column_map[col] = "wind_speed_ms"

        elif "theoretical_power_curve" in low or "theoretical power" in low or ("theoretical" in low and "power" in low):
            column_map[col] = "theoretical_power_ref"

        elif "wind direction" in low or ("direction" in low and "wind" in low):
            column_map[col] = "wind_direction_deg"

    out = out.rename(columns=column_map)

    if "datetime" in out.columns:
        s = out["datetime"].astype(str)
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if dt.isna().mean() > 0.20:
            fallback = pd.to_datetime(s, errors="coerce", format="%d %m %Y %H:%M")
            dt = dt.fillna(fallback)
        out["datetime"] = dt

    for col in ["active_power_kw", "wind_speed_ms", "theoretical_power_ref", "wind_direction_deg"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "datetime" in out.columns:
        out = out.dropna(subset=["datetime"]).sort_values("datetime").copy()

    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"theoretical_power_ref", "active_power_kw"}.issubset(out.columns):
        out["power_gap"] = out["theoretical_power_ref"] - out["active_power_kw"]
        out["capture_ratio"] = np.where(
            out["theoretical_power_ref"].abs() > 0,
            out["active_power_kw"] / out["theoretical_power_ref"],
            np.nan,
        )
    else:
        out["power_gap"] = np.nan
        out["capture_ratio"] = np.nan

    if "datetime" in out.columns:
        out["date"] = out["datetime"].dt.date
        out["hour"] = out["datetime"].dt.hour
        out["month"] = out["datetime"].dt.month
        out["month_label"] = out["datetime"].dt.strftime("%b")
        out["weekday"] = out["datetime"].dt.day_name()

    if "wind_speed_ms" in out.columns:
        out["wind_speed_bin"] = pd.cut(
            out["wind_speed_ms"],
            bins=np.arange(
                np.floor(out["wind_speed_ms"].min(skipna=True)),
                np.ceil(out["wind_speed_ms"].max(skipna=True)) + 0.5,
                0.5,
            ),
            include_lowest=True,
        )

    if "wind_direction_deg" in out.columns:
        out["wind_dir_band"] = (
            (np.floor(pd.to_numeric(out["wind_direction_deg"], errors="coerce") / 30) * 30)
            .astype("Int64")
            .astype(str)
            .replace("<NA>", np.nan)
        )
        out["wind_dir_band"] = out["wind_dir_band"].apply(
            lambda x: f"{x}–{int(x.split('–')[0]) + 30}°" if isinstance(x, str) and "–" not in x else x
        )

    return out


def circular_mode_deg(series: pd.Series, bin_size: int = 10) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "N/A"
    b = (np.floor(s / bin_size) * bin_size).astype(int)
    mode_bin = int(b.value_counts().idxmax())
    return f"{mode_bin}–{mode_bin + bin_size}°"


# =========================================================
# Analytical summaries
# =========================================================
def compute_underperformance(df: pd.DataFrame, quantile_threshold: float) -> Tuple[pd.DataFrame, float]:
    out = df.copy()
    if "power_gap" not in out.columns or out["power_gap"].dropna().empty:
        out["underperforming"] = False
        return out, np.nan

    threshold = float(out["power_gap"].quantile(quantile_threshold))
    out["underperforming"] = out["power_gap"] > threshold
    return out, threshold


def daily_mean_power(df: pd.DataFrame) -> pd.DataFrame:
    if not {"datetime", "active_power_kw"}.issubset(df.columns):
        return pd.DataFrame()
    out = (
        df.set_index("datetime")["active_power_kw"]
        .resample("D")
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={"active_power_kw": "daily_mean_power"})
    )
    return out


def monthly_mean_power(df: pd.DataFrame) -> pd.DataFrame:
    if not {"datetime", "active_power_kw"}.issubset(df.columns):
        return pd.DataFrame()
    out = (
        df.set_index("datetime")["active_power_kw"]
        .resample("MS")
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={"active_power_kw": "monthly_mean_power"})
    )
    return out


def speed_band_summary(df: pd.DataFrame, min_count: int = 40) -> pd.DataFrame:
    needed = {"wind_speed_ms", "active_power_kw", "power_gap", "wind_speed_bin"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    summary = (
        df.dropna(subset=["wind_speed_ms", "active_power_kw", "power_gap", "wind_speed_bin"])
        .groupby("wind_speed_bin", observed=True)
        .agg(
            wind_speed_mean=("wind_speed_ms", "mean"),
            active_power_mean=("active_power_kw", "mean"),
            power_gap_mean=("power_gap", "mean"),
            under_rate=("underperforming", "mean"),
            capture_mean=("capture_ratio", "mean"),
            n=("wind_speed_ms", "size"),
        )
        .reset_index(drop=True)
    )
    summary = summary[summary["n"] >= min_count].copy()
    return summary.sort_values("wind_speed_mean")


def direction_summary(df: pd.DataFrame, min_count: int = 40) -> pd.DataFrame:
    needed = {"wind_dir_band", "active_power_kw"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    summary = (
        df.dropna(subset=["wind_dir_band", "active_power_kw"])
        .groupby("wind_dir_band", observed=True)
        .agg(
            active_power_mean=("active_power_kw", "mean"),
            power_gap_mean=("power_gap", "mean"),
            under_rate=("underperforming", "mean"),
            n=("active_power_kw", "size"),
        )
        .reset_index()
    )
    summary = summary[summary["n"] >= min_count].copy()

    def band_start(x: str) -> int:
        try:
            return int(str(x).split("–")[0])
        except Exception:
            return 9999

    summary["band_start"] = summary["wind_dir_band"].map(band_start)
    summary = summary.sort_values("band_start").drop(columns="band_start")
    return summary


def hour_weekday_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"hour", "weekday", "active_power_kw"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (
        df.pivot_table(
            index="weekday",
            columns="hour",
            values="active_power_kw",
            aggfunc="mean",
        )
        .reindex(weekday_order)
    )
    return pivot


def expected_reference_for_speed(df: pd.DataFrame, wind_speed: float) -> Tuple[float, float]:
    temp = df.copy()
    if not {"wind_speed_ms", "theoretical_power_ref", "active_power_kw"}.issubset(temp.columns):
        return np.nan, np.nan

    temp = temp.dropna(subset=["wind_speed_ms"]).copy()
    temp["speed_bin_center"] = np.round(temp["wind_speed_ms"] * 2) / 2
    band = (
        temp.groupby("speed_bin_center")
        .agg(
            expected_theoretical=("theoretical_power_ref", "median"),
            expected_actual=("active_power_kw", "median"),
        )
        .reset_index()
    )
    if band.empty:
        return np.nan, np.nan

    nearest_idx = (band["speed_bin_center"] - wind_speed).abs().idxmin()
    row = band.loc[nearest_idx]
    return float(row["expected_theoretical"]), float(row["expected_actual"])


def build_data_driven_insights(df: pd.DataFrame) -> List[str]:
    insights: List[str] = []

    avg_power = float(df["active_power_kw"].mean()) if "active_power_kw" in df.columns else np.nan
    avg_ws = float(df["wind_speed_ms"].mean()) if "wind_speed_ms" in df.columns else np.nan
    capture_ratio = (
        safe_divide(float(df["active_power_kw"].sum()), float(df["theoretical_power_ref"].sum()))
        if {"active_power_kw", "theoretical_power_ref"}.issubset(df.columns)
        else np.nan
    )

    insights.append(
        f"The active slice averages {fmt_num(avg_power, 2)} kW of turbine output at {fmt_num(avg_ws, 2)} m/s wind speed, with overall reference capture of {fmt_pct(capture_ratio)}."
    )

    daily = daily_mean_power(df)
    if not daily.empty:
        peak_day = daily.sort_values("daily_mean_power", ascending=False).iloc[0]
        low_day = daily.sort_values("daily_mean_power", ascending=True).iloc[0]
        insights.append(
            f"Daily mean output peaks on {pd.to_datetime(peak_day['datetime']).date()} and is weakest on {pd.to_datetime(low_day['datetime']).date()} within the selected view."
        )

    speed_band = speed_band_summary(df)
    if not speed_band.empty:
        best_band = speed_band.sort_values("capture_mean", ascending=False).iloc[0]
        insights.append(
            f"The strongest power capture appears around {fmt_num(best_band['wind_speed_mean'], 2)} m/s, where average capture ratio is {fmt_pct(best_band['capture_mean'])}."
        )

    if "wind_direction_deg" in df.columns:
        insights.append(
            f"The dominant wind direction band is {circular_mode_deg(df['wind_direction_deg'], bin_size=10)}, indicating the most common operating direction in the current slice."
        )

    month_df = monthly_mean_power(df)
    if not month_df.empty and len(month_df) >= 2:
        peak_month = month_df.sort_values("monthly_mean_power", ascending=False).iloc[0]
        insights.append(
            f"Monthly average output is highest in {pd.to_datetime(peak_month['datetime']).strftime('%b %Y')}."
        )

    return insights[:5]


def build_model_driven_insights(df: pd.DataFrame, threshold_gap: float) -> List[str]:
    insights: List[str] = []

    if "power_gap" not in df.columns or df["power_gap"].dropna().empty:
        return ["Reference-curve benchmark insights are unavailable because the dataset does not contain theoretical power values."]

    under_rate = float(df["underperforming"].mean()) if "underperforming" in df.columns else np.nan
    insights.append(
        f"Underperformance is defined here as the top {fmt_pct(under_rate) if np.isfinite(under_rate) else 'N/A'} of records by power-gap flag share after applying the current quantile threshold, with the gap cut-off at {fmt_num(threshold_gap, 2)}."
    )

    gap_mean = float(df["power_gap"].mean())
    gap_median = float(df["power_gap"].median())
    insights.append(
        f"Across the active slice, the mean power gap is {fmt_num(gap_mean, 2)} and the median power gap is {fmt_num(gap_median, 2)}."
    )

    speed_band = speed_band_summary(df)
    if not speed_band.empty:
        worst_band = speed_band.sort_values("power_gap_mean", ascending=False).iloc[0]
        insights.append(
            f"The largest average performance shortfall occurs around {fmt_num(worst_band['wind_speed_mean'], 2)} m/s, where mean power gap reaches {fmt_num(worst_band['power_gap_mean'], 2)}."
        )

    dir_df = direction_summary(df)
    if not dir_df.empty:
        worst_dir = dir_df.sort_values("under_rate", ascending=False).iloc[0]
        insights.append(
            f"Underperformance is most concentrated in the {worst_dir['wind_dir_band']} direction band."
        )

    if "capture_ratio" in df.columns and df["capture_ratio"].notna().any():
        low_capture_share = float((df["capture_ratio"] < 0.85).mean())
        insights.append(
            f"{fmt_pct(low_capture_share)} of records operate below an 85% actual-to-reference capture ratio in the current view."
        )

    return insights[:5]


# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("Controls")

use_default = st.sidebar.checkbox("Load wind_turbine_scada.csv on startup", value=True)

if use_default:
    if not DEFAULT_DATA.exists():
        st.error("wind_turbine_scada.csv was not found next to app.py. Place it in the same folder or upload a CSV.")
        st.stop()
    raw = load_csv_with_fallback_from_path(str(DEFAULT_DATA))
    source_label = "Loaded: wind_turbine_scada.csv"
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV or enable the default dataset option.")
        st.stop()
    raw = load_csv_with_fallback_from_bytes(uploaded.getvalue())
    source_label = f"Loaded: {uploaded.name}"

df = add_features(standardize_columns(raw))

required = ["datetime", "active_power_kw", "wind_speed_ms"]
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    st.error(
        "The dataset could not be standardised into the minimum required SCADA fields. "
        f"Missing: {', '.join(missing_required)}."
    )
    st.stop()

st.sidebar.caption(source_label)

st.sidebar.divider()
st.sidebar.subheader("Filters")

min_dt = df["datetime"].min()
max_dt = df["datetime"].max()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_dt.date(), max_dt.date()),
    min_value=min_dt.date(),
    max_value=max_dt.date(),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    start_date = min_dt
    end_date = max_dt

ws_min = float(np.nanquantile(df["wind_speed_ms"], 0.01))
ws_max = float(np.nanquantile(df["wind_speed_ms"], 0.99))
wind_speed_range = st.sidebar.slider(
    "Wind speed range (m/s)",
    min_value=float(np.floor(ws_min)),
    max_value=float(np.ceil(ws_max)),
    value=(float(np.floor(ws_min)), float(np.ceil(ws_max))),
)

if "wind_direction_deg" in df.columns and df["wind_direction_deg"].notna().any():
    wd_min = float(np.nanquantile(df["wind_direction_deg"], 0.01))
    wd_max = float(np.nanquantile(df["wind_direction_deg"], 0.99))
    wind_dir_range = st.sidebar.slider(
        "Wind direction range (deg)",
        min_value=float(np.floor(wd_min)),
        max_value=float(np.ceil(wd_max)),
        value=(float(np.floor(wd_min)), float(np.ceil(wd_max))),
    )
else:
    wind_dir_range = None

st.sidebar.divider()
st.sidebar.subheader("Benchmark logic")
quantile_threshold = st.sidebar.slider("Flag top power-gap quantile", 0.70, 0.99, 0.90, step=0.01)
only_under = st.sidebar.checkbox("Show underperforming records only", value=False)

st.sidebar.divider()
st.sidebar.subheader("Visual controls")
scatter_n = st.sidebar.slider("Scatter sample size", 1000, 20000, 5000, step=1000)

# =========================================================
# Apply filters
# =========================================================
base = df.copy()
base = base[(base["datetime"] >= start_date) & (base["datetime"] <= end_date)]
base = base[(base["wind_speed_ms"] >= wind_speed_range[0]) & (base["wind_speed_ms"] <= wind_speed_range[1])]

if wind_dir_range is not None and "wind_direction_deg" in base.columns:
    base = base[(base["wind_direction_deg"] >= wind_dir_range[0]) & (base["wind_direction_deg"] <= wind_dir_range[1])]

dff, threshold_gap = compute_underperformance(base, quantile_threshold=quantile_threshold)

if only_under:
    dff = dff[dff["underperforming"]]

if dff.empty:
    st.warning("No records match the selected filters.")
    st.stop()


# =========================================================
# Hero
# =========================================================
scope_text = (
    f"Source: {source_label} | "
    f"Date range: {dff['datetime'].min().date()} to {dff['datetime'].max().date()} | "
    f"Records in view: {len(dff):,}"
)

hero_text = (
    "This dashboard reframes the SCADA dataset around turbine performance quality: output level, "
    "power-curve alignment, operating-pattern shifts over time, and underperformance identified through "
    "the gap between theoretical and actual power."
)

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">Wind Turbine SCADA Intelligence Dashboard</div>
      <p class="hero-sub">{esc(hero_text)}</p>
      <div class="hero-strip">{esc(scope_text)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# KPIs
# =========================================================
avg_power = float(dff["active_power_kw"].mean())
avg_wind = float(dff["wind_speed_ms"].mean())
median_gap = float(dff["power_gap"].median()) if "power_gap" in dff.columns else np.nan
under_rate = float(dff["underperforming"].mean()) if "underperforming" in dff.columns else np.nan
capture_ratio = (
    safe_divide(float(dff["active_power_kw"].sum()), float(dff["theoretical_power_ref"].sum()))
    if {"active_power_kw", "theoretical_power_ref"}.issubset(dff.columns)
    else np.nan
)
dominant_dir = circular_mode_deg(dff["wind_direction_deg"], bin_size=10) if "wind_direction_deg" in dff.columns else "N/A"

render_kpis(
    [
        ("Average active power", f"{fmt_num(avg_power, 2)} kW", "Mean turbine output in the current slice"),
        ("Average wind speed", f"{fmt_num(avg_wind, 2)} m/s", "Mean wind speed across selected records"),
        ("Reference capture ratio", fmt_pct(capture_ratio), "Actual power as a share of theoretical reference"),
        ("Median power gap", fmt_num(median_gap, 2), "Typical reference-minus-actual difference"),
        ("Underperformance rate", fmt_pct(under_rate), "Share flagged by current gap threshold"),
        ("Dominant wind direction", dominant_dir, "Most common operating direction band"),
    ]
)

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Executive summary",
        "Power curve and efficiency",
        "Time and operating patterns",
        "Underperformance diagnostics",
        "Data appendix",
    ]
)

# =========================================================
# Executive summary
# =========================================================
with tab1:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        render_list_card(
            "1. Data-driven insights",
            build_data_driven_insights(dff),
            "Observed operating patterns based on the filtered SCADA slice.",
        )

        st.write("")
        render_section_title("Daily mean active power")

        daily = daily_mean_power(dff)
        if not daily.empty:
            fig = px.line(
                daily,
                x="datetime",
                y="daily_mean_power",
                labels={"datetime": "Date", "daily_mean_power": "Daily mean active power (kW)"},
            )
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Daily trend needs datetime and active power.")

    with right:
        render_list_card(
            "2. Model-driven insights",
            build_model_driven_insights(dff, threshold_gap),
            "Reference-curve benchmark insights derived from power-gap logic and the underperformance flag.",
        )

        st.write("")
        render_section_title("Operating context snapshot")
        render_badges(
            [
                f"Power-gap threshold: {fmt_num(threshold_gap, 2)}",
                f"Scatter sample size: {scatter_n:,}",
                f"Wind speed filter: {fmt_num(wind_speed_range[0], 1)} to {fmt_num(wind_speed_range[1], 1)} m/s",
            ]
        )

        if "wind_direction_deg" in dff.columns and wind_dir_range is not None:
            render_badges([f"Wind direction filter: {fmt_num(wind_dir_range[0], 0)}° to {fmt_num(wind_dir_range[1], 0)}°"])

        st.write("")
        speed_df = speed_band_summary(dff)
        if not speed_df.empty:
            fig = px.bar(
                speed_df,
                x="wind_speed_mean",
                y="capture_mean",
                labels={"wind_speed_mean": "Wind speed (m/s)", "capture_mean": "Mean capture ratio"},
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Capture-ratio profile needs reference values and enough records per wind-speed band.")


# =========================================================
# Power curve and efficiency
# =========================================================
with tab2:
    render_section_title("Actual power curve against wind speed")
    plot_df = dff.sample(min(scatter_n, len(dff)), random_state=42).copy()

    fig = px.scatter(
        plot_df,
        x="wind_speed_ms",
        y="active_power_kw",
        color="underperforming" if "underperforming" in plot_df.columns else None,
        opacity=0.35,
        labels={"wind_speed_ms": "Wind speed (m/s)", "active_power_kw": "Active power (kW)"},
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="Underperforming")
    st.plotly_chart(fig, use_container_width=True)

    st.write("")
    c1, c2 = st.columns([1.0, 1.0], gap="large")

    with c1:
        render_section_title("Binned actual vs theoretical curve")
        if {"wind_speed_ms", "active_power_kw", "theoretical_power_ref", "wind_speed_bin"}.issubset(dff.columns):
            curve = (
                dff.dropna(subset=["wind_speed_ms", "active_power_kw", "theoretical_power_ref", "wind_speed_bin"])
                .groupby("wind_speed_bin", observed=True)
                .agg(
                    wind_speed_mean=("wind_speed_ms", "mean"),
                    actual_mean=("active_power_kw", "mean"),
                    theoretical_mean=("theoretical_power_ref", "mean"),
                    n=("wind_speed_ms", "size"),
                )
                .reset_index(drop=True)
            )
            curve = curve[curve["n"] >= 40].sort_values("wind_speed_mean")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=curve["wind_speed_mean"],
                    y=curve["actual_mean"],
                    mode="lines+markers",
                    name="Actual mean",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=curve["wind_speed_mean"],
                    y=curve["theoretical_mean"],
                    mode="lines+markers",
                    name="Reference mean",
                )
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Wind speed (m/s)",
                yaxis_title="Power (kW)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This comparison requires actual power, reference power, and wind speed.")

    with c2:
        render_section_title("Efficiency by wind-speed band")
        speed_df = speed_band_summary(dff)
        if not speed_df.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=speed_df["wind_speed_mean"],
                    y=speed_df["capture_mean"],
                    mode="lines+markers",
                    name="Capture ratio",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=speed_df["wind_speed_mean"],
                    y=speed_df["under_rate"],
                    name="Underperformance rate",
                    opacity=0.35,
                    yaxis="y2",
                )
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Wind speed (m/s)",
                yaxis=dict(title="Capture ratio"),
                yaxis2=dict(
                    title="Underperformance rate",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Band-level efficiency needs enough filtered records and reference values.")


# =========================================================
# Time and operating patterns
# =========================================================
with tab3:
    a, b = st.columns([1.2, 0.8], gap="large")

    with a:
        render_section_title("Daily and monthly output")
        daily = daily_mean_power(dff)
        monthly = monthly_mean_power(dff)

        fig = go.Figure()
        if not daily.empty:
            fig.add_trace(
                go.Scatter(
                    x=daily["datetime"],
                    y=daily["daily_mean_power"],
                    mode="lines",
                    name="Daily mean",
                )
            )
        if not monthly.empty:
            fig.add_trace(
                go.Scatter(
                    x=monthly["datetime"],
                    y=monthly["monthly_mean_power"],
                    mode="lines+markers",
                    name="Monthly mean",
                )
            )
        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Date",
            yaxis_title="Active power (kW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with b:
        render_section_title("Wind direction performance")
        dir_df = direction_summary(dff)
        if not dir_df.empty:
            fig = px.bar(
                dir_df,
                x="wind_dir_band",
                y="active_power_mean",
                hover_data=["power_gap_mean", "under_rate", "n"],
                labels={"wind_dir_band": "Wind direction band", "active_power_mean": "Mean active power (kW)"},
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Wind-direction analysis requires wind direction values and enough filtered records.")

    st.write("")
    render_section_title("Weekday-hour operating heatmap")
    heatmap = hour_weekday_heatmap(dff)
    if not heatmap.empty:
        fig = px.imshow(
            heatmap,
            aspect="auto",
            labels=dict(x="Hour of day", y="Weekday", color="Mean active power"),
            text_auto=".0f",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The heatmap requires datetime and active power.")


# =========================================================
# Underperformance diagnostics
# =========================================================
with tab4:
    if "power_gap" not in dff.columns or dff["power_gap"].dropna().empty:
        st.info("Underperformance diagnostics require theoretical reference values.")
    else:
        c1, c2 = st.columns([1.0, 1.0], gap="large")

        with c1:
            render_section_title("Power-gap distribution")
            fig = px.histogram(
                dff.dropna(subset=["power_gap"]),
                x="power_gap",
                nbins=60,
                labels={"power_gap": "Power gap (reference - actual)"},
            )
            fig.add_vline(x=threshold_gap, line_dash="dash")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            render_section_title("Where the biggest gaps occur")
            speed_df = speed_band_summary(dff)
            if not speed_df.empty:
                fig = px.line(
                    speed_df,
                    x="wind_speed_mean",
                    y="power_gap_mean",
                    markers=True,
                    labels={"wind_speed_mean": "Wind speed (m/s)", "power_gap_mean": "Mean power gap"},
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("This chart requires enough filtered records per wind-speed band.")

        st.write("")
        c3, c4 = st.columns([1.0, 1.0], gap="large")

        with c3:
            render_section_title("Direction-based underperformance rate")
            dir_df = direction_summary(dff)
            if not dir_df.empty:
                fig = px.bar(
                    dir_df,
                    x="wind_dir_band",
                    y="under_rate",
                    labels={"wind_dir_band": "Wind direction band", "under_rate": "Underperformance rate"},
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Direction-based diagnostics need wind direction values and enough rows.")

        with c4:
            render_section_title("Top underperforming intervals")
            cols = ["datetime", "wind_speed_ms", "active_power_kw", "theoretical_power_ref", "power_gap"]
            existing = [c for c in cols if c in dff.columns]
            top = dff.sort_values("power_gap", ascending=False).head(15)[existing].copy()
            top = top.rename(columns={"theoretical_power_ref": "reference_power"})
            st.dataframe(top, use_container_width=True, hide_index=True)

        st.write("")
        render_section_title("Scenario checker")
        s1, s2 = st.columns([1.0, 1.0], gap="large")

        with s1:
            scenario_wind = st.number_input(
                "Scenario wind speed (m/s)",
                min_value=0.0,
                max_value=float(np.nanmax(dff["wind_speed_ms"])) if "wind_speed_ms" in dff.columns else 30.0,
                value=float(np.nanmedian(dff["wind_speed_ms"])) if "wind_speed_ms" in dff.columns else 8.0,
                step=0.1,
            )
            scenario_actual = st.number_input(
                "Scenario actual power (kW)",
                min_value=0.0,
                value=float(np.nanmedian(dff["active_power_kw"])) if "active_power_kw" in dff.columns else 0.0,
                step=10.0,
            )

        with s2:
            expected_ref, expected_actual = expected_reference_for_speed(dff, scenario_wind)
            scenario_gap = expected_ref - scenario_actual if np.isfinite(expected_ref) else np.nan
            scenario_flag = bool(np.isfinite(scenario_gap) and np.isfinite(threshold_gap) and scenario_gap > threshold_gap)
            gap_percentile = (
                float((dff["power_gap"].dropna() <= scenario_gap).mean())
                if np.isfinite(scenario_gap) and dff["power_gap"].dropna().shape[0] > 0
                else np.nan
            )

            render_kpis(
                [
                    ("Expected reference power", fmt_num(expected_ref, 2), "Median reference value near the selected wind speed"),
                    ("Typical observed actual power", fmt_num(expected_actual, 2), "Median actual output near the selected wind speed"),
                    ("Scenario power gap", fmt_num(scenario_gap, 2), "Expected reference minus entered actual power"),
                    ("Scenario view", "Flagged" if scenario_flag else "Not flagged", "Compared against current underperformance threshold"),
                    ("Gap percentile", fmt_pct(gap_percentile), "Position of the scenario gap within the current gap distribution"),
                    ("Threshold gap", fmt_num(threshold_gap, 2), "Current underperformance cut-off"),
                ]
            )


# =========================================================
# Data appendix
# =========================================================
with tab5:
    render_section_title("Preparation summary")
    render_badges(
        [
            source_label,
            f"Rows loaded: {len(df):,}",
            f"Rows after filters: {len(dff):,}",
            f"Threshold quantile: {quantile_threshold:.0%}",
        ]
    )

    st.write("")
    a, b = st.columns([1.0, 1.0], gap="large")

    with a:
        render_section_title("Standardised columns found")
        st.dataframe(pd.DataFrame({"column": df.columns}), use_container_width=True, hide_index=True)

    with b:
        render_section_title("Missingness summary")
        miss = dff.isna().mean().sort_values(ascending=False).reset_index()
        miss.columns = ["Column", "Missing share"]
        miss["Missing share"] = miss["Missing share"].map(lambda x: fmt_pct(x))
        st.dataframe(miss, use_container_width=True, hide_index=True)

    st.write("")
    render_section_title("Filtered data preview")
    preview_rows = st.slider("Rows to preview", min_value=20, max_value=200, value=50, step=10)
    st.dataframe(dff.head(preview_rows), use_container_width=True, hide_index=True)

    st.write("")
    csv_bytes = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_bytes,
        file_name="wind_turbine_scada_filtered.csv",
        mime="text/csv",
    )