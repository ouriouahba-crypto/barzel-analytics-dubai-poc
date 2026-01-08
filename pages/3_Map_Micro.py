import streamlit as st
from src.processing.assemble import assemble
from src.app.ui import load_data

st.title("Map & Micro (Analysis)")
pack = assemble(load_data())
f = pack["facts"]

st.map(f.dropna(subset=["latitude","longitude","price_per_sqm"]))
