import streamlit as st
from src.processing.assemble import assemble
from src.app.ui import load_data

st.title("Executive Snapshot (Analysis)")
pack = assemble(load_data())
d = pack["descriptors"]

st.metric("Listings", d["listings"])
st.metric("Median Price / sqm", int(d["price_p50"]) if d["price_p50"] else "n/a")
st.metric("Dispersion (IQR)", int(d["price_iqr"]) if d["price_iqr"] else "n/a")
st.caption("Descriptive analytics only. No decision in app.")
