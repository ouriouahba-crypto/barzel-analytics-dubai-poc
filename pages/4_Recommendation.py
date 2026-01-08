import streamlit as st
from src.processing.assemble import assemble
from src.app.ui import load_data
from src.app.pdf import generate_pdf

st.title("PDF Memo")
pack = assemble(load_data())

if st.button("Generate screening memo (PDF)"):
    path = "/tmp/screening_memo.pdf"
    generate_pdf(path, "Dubai", "Selected scope", pack)
    with open(path, "rb") as f:
        st.download_button("Download PDF", f, file_name="screening_memo.pdf")
