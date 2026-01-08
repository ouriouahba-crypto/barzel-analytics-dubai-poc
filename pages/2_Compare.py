# pages/2_Compare.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text

from src.app.pdf import build_compare_memo_pdf
from src.app.guardrails import require_non_empty, now_utc_str, district_counts
from src.db.db import get_engine
from src.processing.metrics import district_basic_metrics, format_metrics_table, prepare_listings

from src.app.ui import inject_base_ui, hero, pill

inject_base_ui()

DISTRICTS = ["Marina", "Business Bay", "JVC"]


@st.cache_data(ttl=120)
def load_data() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM listings"), conn)

    for col in ["price", "size_sqm", "bedrooms", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


df_all = prepare_listings(load_data())
require_non_empty(df_all, "listings")

counts = district_counts(df_all, DISTRICTS)
hero(
    "Compare",
    "Side-by-side analytics across districts (analysis-only).",
    pills=[
        pill(f"Marina: {counts.get('Marina', 0)}", "good"),
        pill(f"Business Bay: {counts.get('Business Bay', 0)}", "good"),
        pill(f"JVC: {counts.get('JVC', 0)}", "good"),
        pill(f"Refresh: {now_utc_str()}", "warn"),
    ],
)

metrics_raw = district_basic_metrics(df_all, DISTRICTS)
metrics_view = format_metrics_table(metrics_raw)

st.subheader("Metrics table")
st.dataframe(metrics_view, use_container_width=True, hide_index=True)

df = df_all.copy()

st.subheader("Charts")

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(
        metrics_raw,
        x="District",
        y="Weighted Price/sqm (AED)",
        title="Weighted price per sqm (AED/sqm)",
    )
    fig.update_layout(yaxis_title="AED per sqm")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = px.bar(
        metrics_raw,
        x="District",
        y="Median Days Active",
        title="Median Days Active (listing lifetime proxy)",
    )
    fig2.update_layout(yaxis_title="Days")
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.box(
    df.dropna(subset=["price_per_sqm_clip"]),
    x="district",
    y="price_per_sqm_clip",
    points="outliers",
    title="Price per sqm distribution by district (AED/sqm)",
)
fig3.update_layout(xaxis_title="District", yaxis_title="AED per sqm")
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.markdown("### Export")
st.caption("Download a 1-page compare memo (analysis-only).")

pdf_bytes = build_compare_memo_pdf(metrics_df=metrics_raw, df_all=df_all, districts=DISTRICTS)
st.download_button(
    "Download Compare Memo (PDF)",
    data=pdf_bytes,
    file_name="barzel_compare_memo.pdf",
    mime="application/pdf",
)
