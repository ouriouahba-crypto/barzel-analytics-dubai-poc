# src/app/guardrails.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import streamlit as st

# ---------------------------------
# Single source of truth thresholds
# (must match src/processing/engine.py and src/app/pdf.py)
# ---------------------------------
RECO_MIN_N = 10
CONF_MED_N = 10
CONF_HIGH_N = 20


def now_utc_str() -> str:
    """Timezone-aware UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def require_non_empty(df: pd.DataFrame | None, table_name: str = "listings") -> None:
    if df is None or df.empty:
        st.error(f"No data found in the database (table `{table_name}` is empty). Run ingestion first.")
        st.stop()


def require_columns(df: pd.DataFrame, cols: Iterable[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        prefix = f"{context}: " if context else ""
        st.error(f"{prefix}Missing required columns: {', '.join(missing)}")
        st.stop()


def coverage_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return float(df[col].notna().mean())


def confidence_label(n: int) -> str:
    """Return High/Medium/Low based on sample size thresholds."""
    if n >= CONF_HIGH_N:
        return "High"
    if n >= CONF_MED_N:
        return "Medium"
    return "Low"


def can_recommend(n: int) -> bool:
    """Hard gate: do not output decisions under this threshold."""
    return n >= RECO_MIN_N


def warn_low_sample(n: int, label: str = "Sample size") -> str:
    """
    Returns a confidence label and emits a warning/info when needed.
    Uses the same thresholds as the recommendation engine.
    """
    conf = confidence_label(n)

    if conf == "High":
        return conf

    if conf == "Medium":
        st.warning(f"{label} is medium (n={n}). Signals are directional (screening only).")
        return conf

    # Low
    st.warning(f"{label} is low (n={n}). Treat results as weak signal.")
    return conf


def warn_low_coverage(df: pd.DataFrame, col: str, min_ratio: float, label: str | None = None) -> None:
    r = coverage_ratio(df, col)
    if r < min_ratio:
        name = label or col
        st.warning(f"Low coverage for **{name}**: {r*100:.0f}% non-null (min expected {min_ratio*100:.0f}%).")


def district_counts(df_all: pd.DataFrame, districts: list[str]) -> dict[str, int]:
    counts = {d: 0 for d in districts}
    if df_all is None or df_all.empty or "district" not in df_all.columns:
        return counts
    for d in districts:
        counts[d] = int((df_all["district"] == d).sum())
    return counts
