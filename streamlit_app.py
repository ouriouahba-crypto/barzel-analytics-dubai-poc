# streamlit_app.py
import streamlit as st

# MUST be the first Streamlit command in this file
st.set_page_config(page_title="Barzel Analytics – Dubai", layout="wide")

from src.app.ui import inject_base_ui, hero, metric_grid, pill
inject_base_ui()

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

hero(
    "Barzel Analytics — Dubai",
    "Decision-grade screening across Marina / Business Bay / JVC for institutional-style profiles.",
    pills=[
        pill("B2B Demo", "good"),
        pill("Screening, not underwriting", "warn"),
        pill("Dubai POC v0.1", "good"),
    ],
)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("1) Executive Snapshot")
    st.caption("Single decision + KPI recap + 1-page memo.")
    st.markdown("- Profile-based recommendation\n- Consistent KPIs (0–100)\n- PDF export")

with c2:
    st.subheader("2) Compare")
    st.caption("Yield Potential vs Adoption Speed trade-off.")
    st.markdown("- Scatter view\n- Risk-adjusted signal\n- PDF export")

with c3:
    st.subheader("3) Map & Micro")
    st.caption("Spatial view + micro signals with filters.")
    st.markdown("- Map of listings\n- Micro snapshot\n- PDF export")

st.divider()

st.subheader("How to use (30 seconds)")
st.markdown(
    """
1. Go to **Executive Snapshot** (sidebar) and pick an investor profile.  
2. Read the decision + KPI recap.  
3. Download the **1-page PDF memo** from the page.  
4. Use **Compare** for trade-off confirmation, and **Map & Micro** for micro sanity-check.
"""
)

st.subheader("Methodology (screening)")
metric_grid(
    [
        ("Yield Potential", "0–100", "Value proxy from median price/sqm (not rental yield)"),
        ("Adoption Speed", "0–100", "Liquidity proxy from DOM + relative volume share"),
        ("Risk Index", "0–100", "Dispersion proxy (higher = more risky)"),
        ("Barzel Score", "0–100", "Composite; risk inverted inside the score"),
    ]
)

st.info("Navigation: use the left sidebar pages. This is a screening demo — not underwriting, pricing, or investment advice.")
