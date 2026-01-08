# src/processing/metrics.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


def _to_datetime(s: pd.Series) -> pd.Series:
    """Parse datetimes robustly (UTC-aware when possible)."""
    return pd.to_datetime(s, errors="coerce", utc=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used across the dashboard.

    This repo uses a listings dataset (not transactions). Some KPIs are *proxies*:
    - price_per_sqm: price / size_sqm
    - days_active: (coalesce(last_seen, now) - first_seen) in days (listing lifetime proxy)

    Notes:
    - last_seen is often missing in scraped datasets; we treat missing last_seen as "still active today".
    - negative durations are set to NaN.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Safe numerics
    if "price" in out.columns:
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
    if "size_sqm" in out.columns:
        out["size_sqm"] = pd.to_numeric(out["size_sqm"], errors="coerce")

    # Safe price_per_sqm
    if "price" in out.columns and "size_sqm" in out.columns:
        price = out["price"]
        sqm = out["size_sqm"]
        out["price_per_sqm"] = np.where((price > 0) & (sqm > 0), price / sqm, np.nan)

    # Listing lifetime proxy (days_active)
    if "first_seen" in out.columns:
        fs = _to_datetime(out["first_seen"])
        if "last_seen" in out.columns:
            ls = _to_datetime(out["last_seen"])
        else:
            ls = pd.Series([pd.NaT] * len(out), index=out.index)

        now_utc = pd.Timestamp.utcnow()

        ls = ls.fillna(now_utc)

        days = (ls - fs).dt.total_seconds() / 86400.0
        out["days_active"] = days

        # Clean impossible values
        out.loc[out["days_active"] < 0, "days_active"] = np.nan
        # Cap absurd listings (keeps charts readable)
        out.loc[out["days_active"] > 3650, "days_active"] = np.nan  # >10y

    return out

# -------------------------
# Cleaning + visualization helpers
# -------------------------

def _clip_series_quantiles(s: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    lo, hi = s.quantile([q_low, q_high])
    return lo, hi


def prepare_listings(
    df: pd.DataFrame,
    *,
    ppsqm_min: float = 1000.0,
    ppsqm_max: float = 25000.0,
    clip_q_low: float = 0.05,
    clip_q_high: float = 0.95,
) -> pd.DataFrame:
    """Standardize -> derive -> clean.

    Output guarantees:
    - price_per_sqm exists (float)
    - price_per_sqm_clip exists (float) for charts/maps
    - days_active exists (float) when first_seen is available

    The dataset is *listings*, so we remove obviously-invalid rows for pricing analytics.
    """
    if df is None or df.empty:
        return df

    out = add_derived_columns(df)

    # Hard sanity filter for price_per_sqm (removes division artifacts / wrong units)
    if "price_per_sqm" in out.columns:
        out["price_per_sqm"] = pd.to_numeric(out["price_per_sqm"], errors="coerce")
        out.loc[(out["price_per_sqm"] < ppsqm_min) | (out["price_per_sqm"] > ppsqm_max), "price_per_sqm"] = np.nan

        # Quantile clip for visualization stability
        valid = out["price_per_sqm"].dropna()
        if not valid.empty:
            lo, hi = valid.quantile([clip_q_low, clip_q_high])
            out["price_per_sqm_clip"] = out["price_per_sqm"].clip(lo, hi)
        else:
            out["price_per_sqm_clip"] = np.nan
    else:
        out["price_per_sqm_clip"] = np.nan

    # Clean days_active (keep only plausible values for charts)
    if "days_active" in out.columns:
        out["days_active"] = pd.to_numeric(out["days_active"], errors="coerce")
        out.loc[(out["days_active"] < 0) | (out["days_active"] > 3650), "days_active"] = np.nan

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

        src_col = "price_per_sqm_clip" if "price_per_sqm_clip" in ddf.columns else "price_per_sqm"
        med_ppsqm = float(np.nanmedian(ddf.get(src_col, pd.Series(dtype=float)))) if listings else np.nan
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
