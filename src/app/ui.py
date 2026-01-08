import pandas as pd
import streamlit as st
from pathlib import Path

# =========================================================
# Base UI helpers (safe, neutral, premium)
# =========================================================
def inject_base_ui():
    pass


def hero(title: str, subtitle: str = ""):
    st.markdown(f"""
        <div style="padding:1.5rem 0;">
            <h1 style="margin-bottom:0.2rem;">{title}</h1>
            <p style="color:#6b7280;font-size:1.05rem;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def metric_grid(metrics: list):
    cols = st.columns(len(metrics))
    for col, (label, value, help_text) in zip(cols, metrics):
        with col:
            st.metric(label, value, help=help_text)

def pill(text: str, status: str = "neutral"):
    colors = {
        "good": ("#ecfdf5", "#065f46"),
        "neutral": ("#eef2ff", "#3730a3"),
        "warn": ("#fffbeb", "#92400e"),
        "bad": ("#fef2f2", "#991b1b"),
    }
    bg, fg = colors.get(status, colors["neutral"])

    st.markdown(
        f"<span style='background:{bg};color:{fg};"
        f"padding:4px 10px;border-radius:999px;"
        f"font-size:0.8rem;margin-right:6px;'>{text}</span>",
        unsafe_allow_html=True,
    )


# =========================================================
# Data loader (centralised)
# =========================================================

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    DATA_PATH = Path("data/listings_enriched.csv")

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    # minimal normalization
    for col in ["district", "city", "property_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df
