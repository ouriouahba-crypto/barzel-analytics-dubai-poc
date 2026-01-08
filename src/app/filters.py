# src/app/filters.py
from __future__ import annotations

import pandas as pd
import streamlit as st


def sidebar_transaction_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Sidebar filter to avoid mixing incompatible record types."""
    if df is None or df.empty or "transaction_type" not in df.columns:
        return df, "Listings"

    st.sidebar.subheader("Data scope")
    mode = st.sidebar.radio(
        "Transaction type",
        ["Sale (buy)", "Rent (contracts)", "All"],
        index=0,
        key="tx_mode",
    )

    tt = df["transaction_type"].astype(str).str.lower().str.strip()
    if mode.startswith("Sale"):
        df = df[tt == "sale"].copy()
    elif mode.startswith("Rent"):
        df = df[tt == "rent"].copy()

    return df, mode
