# streamlit_app.py
import streamlit as st

# MUST be the first Streamlit command in this file
st.set_page_config(page_title="Barzel Analytics – Dubai", layout="wide")

from src.app.ui import inject_base_ui, hero, metric_grid, pill

inject_base_ui()

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

hero(
    "Barzel Analytics — Dubai",
    "Institutional-style market analytics across Marina / Business Bay / JVC using listing-level proxy data.",
    pills=[
        pill("Analysis-only UI", "good"),
        pill("Screening proxies", "warn"),
        pill("PDF memo = narrative", "good"),
    ],
)

st.subheader("What this demo is")
st.write(
    "A compact market analytics cockpit. The UI shows **descriptive metrics and distributions** only. "
    "It does not display a final ranking or conclusion."
)

st.subheader("What the PDF memo is")
st.write(
    "If you need a committee-friendly narrative (with Barzel synthesis and gated conclusion), use the **PDF Memo Builder** page. "
    "That keeps the dashboard compliant-friendly while still allowing a professional memo output."
)

st.subheader("Dataset & proxies")
metric_grid(
    [
        ("Price per sqm", "AED/sqm", "Computed from price / size_sqm (where available)"),
        ("Days Active", "days", "Listing lifetime proxy: last_seen - first_seen"),
        ("Dispersion", "proxy", "Used internally as a risk proxy (not a guarantee)"),
        ("Scope", "3 districts", "Marina / Business Bay / JVC (demo scope)"),
    ]
)

st.info(
    "Navigation: use the left sidebar pages. "
    "This is a screening demo — not underwriting, valuation, or investment advice."
)
