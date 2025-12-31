# streamlit_app.py
import streamlit as st

# MUST be the first Streamlit command in this file
st.set_page_config(page_title="Barzel Analytics – Dubai", layout="wide")

from src.app.ui import inject_base_ui, hero, metric_grid, pill

inject_base_ui()

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

hero(
    "Barzel Analytics — Dubai",
    "Institutional-style district screening across Marina / Business Bay / JVC (proxy KPIs, 0–100).",
    pills=[
        pill("B2B Demo", "good"),
        pill("Screening, not underwriting", "warn"),
        pill("Simulated mandate (POC)", "warn"),
        pill("Dubai POC v0.1", "good"),
    ],
)

st.caption(
    "This demo is designed for **first-pass allocation screening**. "
    "It is **not** underwriting, pricing, valuation, or investment advice."
)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("1) Executive Snapshot")
    st.caption("Single decision + KPI recap + memo-ready thesis.")
    st.markdown(
        "- Decision + confidence gate\n"
        "- KPI recap (0–100 proxies)\n"
        "- 1-page PDF memo"
    )

with c2:
    st.subheader("2) Compare")
    st.caption("Trade-off view (Value proxy vs Liquidity proxy) with risk-awareness.")
    st.markdown(
        "- IC-style trade-off summary\n"
        "- Scatter + ranking table\n"
        "- 1-page PDF memo"
    )

with c3:
    st.subheader("3) Map & Micro")
    st.caption("Micro sanity-check (filters) to detect outliers / dispersion pockets.")
    st.markdown(
        "- Spatial distribution\n"
        "- Micro signals (descriptive)\n"
        "- 1-page PDF memo"
    )

st.divider()

st.subheader("How to use (30 seconds)")
st.markdown(
    """
1. Open **Executive Snapshot** and pick an investor profile.  
2. Read the **Decision + Thesis** and check **Confidence / sample size**.  
3. Download the **1-page PDF memo**.  
4. Use **Compare** to validate the trade-off (value vs liquidity).  
5. Use **Map & Micro** to sanity-check micro distribution (outliers, dispersion pockets).
"""
)

st.subheader("Methodology (screening proxies)")
metric_grid(
    [
        ("Value Proxy", "0–100", "Derived from median price/sqm (lower AED/sqm ⇒ higher score)"),
        ("Liquidity Proxy", "0–100", "Adoption Speed from DOM + relative volume share"),
        ("Risk Index", "0–100", "Dispersion proxy (higher = more risky)"),
        ("Barzel Score", "0–100", "Composite (risk inverted inside the score)"),
    ]
)

st.info(
    "Navigation: use the left sidebar pages. "
    "This is a screening demo — not underwriting, pricing, valuation, or investment advice."
)
