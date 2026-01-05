# src/processing/metrics.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used across the dashboard.

    This repo uses a listing dataset (not transactions). Some KPIs are *proxies*:
    - price_per_sqm: price / size_sqm
    - days_active: last_seen - first_seen (listing lifetime proxy)
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Safe price_per_sqm
    if "price" in out.columns and "size_sqm" in out.columns:
        price = pd.to_numeric(out["price"], errors="coerce")
        sqm = pd.to_numeric(out["size_sqm"], errors="coerce")
        out["price_per_sqm"] = np.where((price > 0) & (sqm > 0), price / sqm, np.nan)

    # Listing lifetime proxy
    if "first_seen" in out.columns and "last_seen" in out.columns:
        fs = _to_datetime(out["first_seen"])
        ls = _to_datetime(out["last_seen"])
        out["days_active"] = (ls - fs).dt.days

    return out


def district_basic_metrics(df_all: pd.DataFrame, districts: list[str]) -> pd.DataFrame:
    """Compute a compact, dashboard-friendly metrics table per district."""
    if df_all is None or df_all.empty:
        return pd.DataFrame(
            {
                "District": districts,
                "Listings": [0] * len(districts),
                "Median Price (AED)": [np.nan] * len(districts),
                "Median Size (sqm)": [np.nan] * len(districts),
                "Weighted Price/sqm (AED)": [np.nan] * len(districts),
                "Median Price/sqm (AED)": [np.nan] * len(districts),
                "Median Days Active": [np.nan] * len(districts),
            }
        )

    df = add_derived_columns(df_all)

    rows = []
    for d in districts:
        ddf = df[df.get("district") == d].copy()

        listings = int(len(ddf))
        med_price = float(np.nanmedian(ddf.get("price", pd.Series(dtype=float)))) if listings else np.nan
        med_sqm = float(np.nanmedian(ddf.get("size_sqm", pd.Series(dtype=float)))) if listings else np.nan

        # Weighted price per sqm: sum(price)/sum(sqm)
        price = pd.to_numeric(ddf.get("price", pd.Series(dtype=float)), errors="coerce")
        sqm = pd.to_numeric(ddf.get("size_sqm", pd.Series(dtype=float)), errors="coerce")
        denom = float(np.nansum(sqm)) if listings else 0.0
        w_ppsqm = float(np.nansum(price) / denom) if denom > 0 else np.nan

        med_ppsqm = float(np.nanmedian(ddf.get("price_per_sqm", pd.Series(dtype=float)))) if listings else np.nan
        med_days = float(np.nanmedian(ddf.get("days_active", pd.Series(dtype=float)))) if listings else np.nan

        rows.append(
            {
                "District": d,
                "Listings": listings,
                "Median Price (AED)": med_price,
                "Median Size (sqm)": med_sqm,
                "Weighted Price/sqm (AED)": w_ppsqm,
                "Median Price/sqm (AED)": med_ppsqm,
                "Median Days Active": med_days,
            }
        )

    return pd.DataFrame(rows)


def format_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Human-friendly formatting for Streamlit display (keeps raw values in the PDF layer)."""
    out = df.copy()

    int_cols = ["Listings"]
    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    money_cols = ["Median Price (AED)"]
    for c in money_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")

    float_cols = ["Median Size (sqm)", "Weighted Price/sqm (AED)", "Median Price/sqm (AED)", "Median Days Active"]
    for c in float_cols:
        if c in out.columns:
            if "Price/sqm" in c:
                out[c] = out[c].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
            else:
                out[c] = out[c].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "—")

    return out
