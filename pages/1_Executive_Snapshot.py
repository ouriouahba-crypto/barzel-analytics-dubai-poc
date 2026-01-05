# pages/1_Executive_Snapshot.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text

from src.app.pdf import build_executive_memo_pdf
from src.app.guardrails import require_non_empty, now_utc_str, district_counts, warn_low_coverage
from src.db.db import get_engine
from src.processing.metrics import district_basic_metrics, format_metrics_table, add_derived_columns

from src.app.ui import inject_base_ui, hero, metric_grid, pill

inject_base_ui()

DISTRICTS = ["Marina", "Business Bay", "JVC"]


@st.cache_data(ttl=120)
def load_data() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM listings"), conn)

    # Normalize numeric columns
    for col in ["price", "size_sqm", "bedrooms", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


df_all = load_data()
require_non_empty(df_all, "listings")

# Basic coverage checks (dataset-specific)
warn_low_coverage(df_all, "size_sqm", min_ratio=0.80, label="Size (sqm)")
warn_low_coverage(df_all, "price", min_ratio=0.95, label="Price")

counts = district_counts(df_all, DISTRICTS)
total_n = int(len(df_all))

hero(
    "Market Overview",
    "Descriptive analytics across Marina / Business Bay / JVC using listing-level proxy data (no on-screen conclusion).",
    pills=[
        pill(f"Listings: {total_n}", "good"),
        pill(f"Marina: {counts.get('Marina', 0)}", "good"),
        pill(f"Business Bay: {counts.get('Business Bay', 0)}", "good"),
        pill(f"JVC: {counts.get('JVC', 0)}", "good"),
        pill(f"Refresh: {now_utc_str()}", "warn"),
    ],
)

st.caption(
    "This dashboard is **analysis-only**. It does not display a final ranking or conclusion. "
    "For a committee-style narrative memo, use the PDF Memo Builder page."
)

# -------------------------
# Core metrics table
# -------------------------
metrics_raw = district_basic_metrics(df_all, DISTRICTS)
metrics_view = format_metrics_table(metrics_raw)

st.subheader("Key metrics (per district)")
st.dataframe(metrics_view, use_container_width=True, hide_index=True)

# -------------------------
# Distribution views
# -------------------------
st.subheader("Distributions")

df = add_derived_columns(df_all)

left, right = st.columns(2)

with left:
    fig = px.box(
        df.dropna(subset=["price_per_sqm"]),
        x="district",
        y="price_per_sqm",
        points="outliers",
        title="Price per sqm distribution (AED/sqm)",
    )
    fig.update_layout(xaxis_title="District", yaxis_title="AED per sqm")
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig2 = px.histogram(
        df.dropna(subset=["days_active"]),
        x="days_active",
        color="district",
        barmode="overlay",
        nbins=30,
        title="Days Active (listing lifetime proxy)",
    )
    fig2.update_layout(xaxis_title="Days", yaxis_title="Count")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# PDF export (analysis-only)
# -------------------------
st.divider()
st.markdown("### Export")
st.caption("Download a 1-page *analytical* memo for sharing (no recommendation in the document).")

pdf_bytes = build_executive_memo_pdf(metrics_df=metrics_raw, df_all=df_all, districts=DISTRICTS)
st.download_button(
    "Download Executive Data Summary (PDF)",
    data=pdf_bytes,
    file_name="barzel_executive_data_summary.pdf",
    mime="application/pdf",
)
