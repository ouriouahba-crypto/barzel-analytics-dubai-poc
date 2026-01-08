import streamlit as st
from src.processing.assemble import assemble
from src.app.ui import load_data

st.title("Compare (Analysis)")
pack = assemble(load_data())
f = pack["facts"]

st.bar_chart(f.groupby("district")["price_per_sqm"].median())
st.bar_chart(f.groupby("district")["days_active"].median())
