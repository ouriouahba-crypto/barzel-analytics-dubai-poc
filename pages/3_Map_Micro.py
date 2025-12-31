# pages/3_Map_Micro.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import text

from src.app.pdf import build_micro_memo_pdf
from src.app.guardrails import (
    require_non_empty,
    now_utc_str,
    district_counts,
    warn_low_coverage,
    warn_low_sample,
)
from src.processing.kpis import adoption_speed_score
from src.db.db import get_engine

from src.app.ui import inject_base_ui, hero, metric_grid, pill

inject_base_ui()

DISTRICTS = ["Marina", "Business Bay", "JVC"]


@st.cache_data(ttl=120)
def load_data() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM listings"), conn)

    for col in ["price", "size_sqm", "latitude", "longitude", "bedrooms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _ppsqm_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if "price" not in df.columns or "size_sqm" not in df.columns:
        return pd.Series(dtype="float64")
    tmp = df[["price", "size_sqm"]].dropna()
    tmp = tmp[tmp["size_sqm"] > 0]
    if tmp.empty:
        return pd.Series(dtype="float64")
    return (tmp["price"] / tmp["size_sqm"]).replace([np.inf, -np.inf], np.nan).dropna()


def _confidence_bucket(n: int) -> str:
    # Aligned with engine.py
    if n >= 20:
        return "High"
    if n >= 10:
        return "Medium"
    return "Low"


df_all = load_data()
require_non_empty(df_all, "listings")

warn_low_coverage(df_all, "latitude", 0.40, "latitude")
warn_low_coverage(df_all, "longitude", 0.40, "longitude")
warn_low_coverage(df_all, "price", 0.70, "price")
warn_low_coverage(df_all, "size_sqm", 0.60, "size_sqm")

# Sidebar filters
st.sidebar.header("Filters")
district = st.sidebar.selectbox("District", ["All"] + DISTRICTS, index=0)

bedroom_options: list[int] = []
if "bedrooms" in df_all.columns:
    raw = df_all["bedrooms"].dropna().unique()
    cleaned = []
    for x in raw:
        try:
            fx = float(x)
            if fx.is_integer():
                cleaned.append(int(fx))
        except Exception:
            pass
    bedroom_options = sorted(set(cleaned))

bedrooms = st.sidebar.multiselect("Bedrooms", bedroom_options)

# Map color settings (simple + useful)
st.sidebar.divider()
st.sidebar.subheader("Map color")
cap_quantiles = st.sidebar.checkbox("Stabilize colors (cap 5%–95%)", value=True)
# Red = expensive (default): YlOrRd (yellow low -> red high)
reverse_scale = st.sidebar.checkbox("Reverse scale (red = cheaper)", value=False)

df = df_all.copy()
if district != "All" and "district" in df.columns:
    df = df[df["district"] == district]
if bedrooms and "bedrooms" in df.columns:
    df = df[df["bedrooms"].isin(bedrooms)]

n = int(len(df))
conf = _confidence_bucket(n)
warn_low_sample(n, "Filtered sample size")

hero(
    "Map & Micro",
    "Micro sanity-check to validate distribution (outliers, dispersion pockets). Screening only.",
    pills=[
        pill(f"Confidence: {conf}", "good" if conf == "High" else ("warn" if conf == "Medium" else "bad")),
        pill("Descriptive only (not underwriting)", "warn"),
        pill(f"Refresh: {now_utc_str()}", "good"),
    ],
)

st.caption(
    f"Sample size: **{n}** listings "
    f"(District: **{district}**, Bedrooms: **{bedrooms if bedrooms else 'All'}**)"
)

st.info(
    "Purpose: this page is **not** the decision engine. "
    "It is a **micro sanity-check** to detect outliers and distribution issues before moving to underwriting."
)

# Global benchmark
g_ppsqm = _ppsqm_series(df_all)
q_global = g_ppsqm.quantile([0.25, 0.50, 0.75]).to_dict() if not g_ppsqm.empty else None

median_price = float(df["price"].median()) if (n > 0 and "price" in df.columns) else 0.0
median_size = float(df["size_sqm"].median()) if (n > 0 and "size_sqm" in df.columns) else 0.0

ppsqm_local = _ppsqm_series(df)
median_ppsqm = float(ppsqm_local.median()) if not ppsqm_local.empty else 0.0

liq_local = None
if district != "All" and n > 0:
    liq_local = float(adoption_speed_score(df, df_all))

iqr_val = None
if not ppsqm_local.empty and len(ppsqm_local) >= 5:
    q25, q75 = ppsqm_local.quantile([0.25, 0.75])
    iqr_val = float(q75 - q25)

pct_val = None
if not ppsqm_local.empty and not g_ppsqm.empty:
    local_median = float(ppsqm_local.median())
    pct_val = float((g_ppsqm <= local_median).mean() * 100.0)


def pricing_level_call(pct: float | None) -> str:
    if pct is None:
        return "pricing level: unknown"
    if pct >= 75:
        return "pricing level: premium"
    if pct >= 40:
        return "pricing level: mid-market"
    return "pricing level: value"


def dispersion_call(iqr: float | None, _qg: dict | None) -> str:
    if iqr is None:
        return "dispersion: unknown"
    if iqr <= 2500:
        return "dispersion: tight"
    if iqr <= 6000:
        return "dispersion: moderate"
    return "dispersion: wide"


def liquidity_call(liq: float | None) -> str:
    if liq is None:
        return "liquidity: unknown"
    if liq >= 70:
        return "liquidity: high (proxy)"
    if liq >= 40:
        return "liquidity: medium (proxy)"
    return "liquidity: low (proxy)"


micro_call = [pricing_level_call(pct_val), dispersion_call(iqr_val, q_global), liquidity_call(liq_local)]

if conf == "Low":
    st.warning("Micro read is **too thin** under current filters. We show descriptive stats, but avoid strong calls.")
    st.info("Tip: remove filters or broaden the bedroom range to increase n.")
else:
    st.success(f"Micro read: **{micro_call[0]}, {micro_call[1]}, {micro_call[2]}** (screening only)")

# -------------------------
# Map
# -------------------------
st.subheader("Listings Map")
if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
    st.info("No listings to display with the current filters.")
else:
    map_df = df.dropna(subset=["latitude", "longitude"]).copy()
    map_df = map_df[(map_df["latitude"].between(24.8, 25.5)) & (map_df["longitude"].between(55.0, 55.6))]
    if map_df.empty:
        st.info("No valid coordinates available for the current filters.")
    else:
        map_df["price_per_sqm"] = np.nan
        if "price" in map_df.columns and "size_sqm" in map_df.columns:
            mask = map_df["price"].notna() & map_df["size_sqm"].notna() & (map_df["size_sqm"] > 0)
            map_df.loc[mask, "price_per_sqm"] = map_df.loc[mask, "price"] / map_df.loc[mask, "size_sqm"]

        # ---- Color stabilization (prevents outliers from ruining the scale)
        valid_ppsqm = map_df["price_per_sqm"].replace([np.inf, -np.inf], np.nan).dropna()

        lo, hi = None, None
        if not valid_ppsqm.empty:
            if cap_quantiles:
                lo = float(valid_ppsqm.quantile(0.05))
                hi = float(valid_ppsqm.quantile(0.95))
            else:
                lo = float(valid_ppsqm.min())
                hi = float(valid_ppsqm.max())

        map_df["price_per_sqm_capped"] = map_df["price_per_sqm"]
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            map_df["price_per_sqm_capped"] = map_df["price_per_sqm"].clip(lower=lo, upper=hi)

        # ---- Color scale (yellow -> red), option to reverse
        scale = px.colors.sequential.YlOrRd
        if reverse_scale:
            scale = list(reversed(scale))

        fig = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="price_per_sqm_capped",
            color_continuous_scale=scale,
            range_color=(lo, hi) if (lo is not None and hi is not None and hi > lo) else None,
            hover_name="district" if "district" in map_df.columns else None,
            hover_data={
                "price": "price" in map_df.columns,
                "size_sqm": "size_sqm" in map_df.columns,
                "bedrooms": "bedrooms" in map_df.columns,
                "price_per_sqm": True,            # ✅ true value
                "price_per_sqm_capped": False,    # hide internal cap column
                "latitude": False,
                "longitude": False,
            },
            zoom=11,
            height=520,
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="AED / sqm"),
        )
        st.plotly_chart(fig, use_container_width=True)

st.caption("Map color = price per sqm (value / premium proxy). Colors are stabilized to avoid outlier distortion.")

# -------------------------
# Snapshot
# -------------------------
st.subheader("Micro Market Snapshot")
metric_grid(
    [
        ("Listings", str(n), None),
        ("Median Price", f"{median_price:,.0f} AED", None),
        ("Median Price / sqm", f"{median_ppsqm:,.0f} AED/sqm", "Value proxy"),
        ("Median Size", f"{median_size:,.0f} sqm", None),
    ]
)

# -------------------------
# Histogram
# -------------------------
st.subheader("Price Distribution (Histogram)")
if n == 0 or "price" not in df.columns:
    st.info("No price data available for the current filters.")
else:
    price_series = df["price"].replace([np.inf, -np.inf], np.nan).dropna()
    if price_series.empty:
        st.info("No valid price values available.")
    else:
        lo_h, hi_h = price_series.quantile([0.02, 0.98])
        clipped = price_series.clip(lower=float(lo_h), upper=float(hi_h))
        hist = px.histogram(
            clipped,
            nbins=30,
            title="Histogram of Prices (winsorized 2%–98%)",
            labels={"value": "Price (AED)", "count": "Listings"},
        )
        hist.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(hist, use_container_width=True)

st.caption("Histogram is winsorized (2%–98%) to reduce outlier impact.")

# -------------------------
# Micro signals
# -------------------------
st.subheader("Micro Signals")
sig1, sig2, sig3, sig4 = st.columns(4)

with sig1:
    st.metric("Confidence", conf, help="Based on filtered sample size (n).")
    st.caption(f"n = {n}")

with sig2:
    if liq_local is None:
        st.metric("Liquidity (local)", "—")
        st.caption("Select a single district")
    else:
        st.metric("Liquidity (local)", f"{liq_local:.1f}")
        st.caption("Adoption Speed proxy")

with sig3:
    if iqr_val is None:
        st.metric("Pricing tightness", "—")
        st.caption("Not enough data")
    else:
        st.metric("Pricing tightness", f"{iqr_val:,.0f} AED/sqm")
        st.caption("IQR (lower = tighter)")

with sig4:
    if pct_val is None:
        st.metric("Market price percentile", "—")
        st.caption("No global benchmark")
    else:
        st.metric("Market price percentile", f"P{pct_val:.0f}")
        st.caption("Percentile vs market (lower = cheaper)")

# -------------------------
# Micro Summary (so-what)
# -------------------------
st.markdown("### Micro Summary (screening)")
pct_display = f"P{pct_val:.0f}" if pct_val is not None else "no benchmark"
iqr_display = f"{iqr_val:,.0f}" if iqr_val is not None else "—"
liq_display = f"{liq_local:.1f}" if liq_local is not None else "—"

if conf == "Low":
    st.write("• **Low confidence** under current filters — treat as descriptive only.")
else:
    st.write(
        f"• Pricing indicates **{micro_call[0].split(': ')[1]}** ({pct_display}) with "
        f"**{micro_call[1].split(': ')[1]}** dispersion (IQR {iqr_display} AED/sqm)."
    )

if district == "All":
    st.write("• Select a single district to compute a cleaner local liquidity proxy (Adoption Speed).")
else:
    st.write(f"• Liquidity proxy for **{district}** is **{micro_call[2].split(': ')[1]}** (Adoption Speed {liq_display}).")

st.caption("Micro Summary is descriptive only and does not drive the Recommendation engine.")

# -------------------------
# Data provenance + Export
# -------------------------
st.divider()
st.markdown("### Data provenance")
counts = district_counts(df_all, DISTRICTS)
st.caption(
    f"Data refreshed: **{now_utc_str()}** · Total listings: **{len(df_all)}** · "
    f"Marina: **{counts['Marina']}** | Business Bay: **{counts['Business Bay']}** | JVC: **{counts['JVC']}**"
)

st.divider()
st.markdown("### Export")

pdf_bytes = build_micro_memo_pdf(
    district=district,
    bedrooms=bedrooms,
    n=n,
    conf=conf,
    median_price=median_price,
    median_ppsqm=median_ppsqm,
    median_size=median_size,
    liq_local=liq_local,
    pct_val=pct_val,
    iqr_val=iqr_val,
    df_all=df_all,
)

safe_district = district.lower().replace(" ", "_").replace("-", "_")
safe_bed = "all" if not bedrooms else "b" + "-".join(str(x) for x in bedrooms)

st.download_button(
    label="📄 Download 1-page Micro Memo (PDF)",
    data=pdf_bytes,
    file_name=f"barzel_micro_memo_{safe_district}_{safe_bed}.pdf",
    mime="application/pdf",
)
