import pandas as pd
import streamlit as st
from pathlib import Path

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Centralized data loader for Barzel Analytics V2.
    Reads the enriched listings dataset.
    """

    # 👉 adapte si ton CSV a un autre nom
    DATA_PATH = Path("data/listings_enriched.csv")

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    # Normalisation minimale (neutre, non décisionnelle)
    for col in ["district", "city", "property_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df
