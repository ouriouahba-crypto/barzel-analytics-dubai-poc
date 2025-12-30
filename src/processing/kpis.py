# src/processing/kpis.py
from __future__ import annotations

import pandas as pd
import numpy as np

SCORE_MIN = 0.0
SCORE_MAX = 100.0

YIELD_PPSQM_BEST = 10_000.0
YIELD_PPSQM_WORST = 40_000.0

ADOPTION_W_DOM = 0.60
ADOPTION_W_VOL = 0.40

# If DOM is unavailable, we reduce confidence in "adoption speed" (liquidity proxy)
ADOPTION_NO_DOM_PENALTY = 0.75  # 25% penalty


def _clip01_100(x: float) -> float:
    return float(np.clip(float(x), SCORE_MIN, SCORE_MAX))


# -------------------------
# Adoption Speed (Liquidity proxy) - higher = better
# -------------------------
def adoption_speed_score(df: pd.DataFrame, all_df: pd.DataFrame | None = None) -> float:
    if df is None or df.empty:
        return 0.0

    d = df.copy()

    dom_score = None
    if "first_seen" in d.columns and "last_seen" in d.columns:
        d["first_seen"] = pd.to_datetime(d["first_seen"], errors="coerce")
        d["last_seen"] = pd.to_datetime(d["last_seen"], errors="coerce")

        dom = (d["last_seen"] - d["first_seen"]).dt.days
        if dom.notna().sum() >= 3:
            med = float(dom.dropna().median())
            med = max(med, 0.0)
            dom_score = 100.0 - min(med, 100.0)

    # Volume score (0-100)
    if all_df is not None and isinstance(all_df, pd.DataFrame) and len(all_df) > 0:
        volume_ratio = len(d) / len(all_df)
        volume_score = min(volume_ratio * 200.0, 100.0)  # 50% share => 100
    else:
        volume_score = min(len(d) * 5.0, 100.0)

    if dom_score is None:
        # Pure volume is not true liquidity; penalize to avoid misleading "liquidity" spikes
        score = ADOPTION_NO_DOM_PENALTY * float(volume_score)
    else:
        score = (ADOPTION_W_DOM * float(dom_score)) + (ADOPTION_W_VOL * float(volume_score))

    return round(_clip01_100(score), 1)


# -------------------------
# Risk Index (dispersion proxy) - higher = MORE risky (worse)
# Use price per sqm when possible to avoid product-mix bias.
# -------------------------
def risk_index_score(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0

    series = None

    if "price" in df.columns and "size_sqm" in df.columns:
        price = df["price"].replace([np.inf, -np.inf], np.nan)
        size = df["size_sqm"].replace([np.inf, -np.inf], np.nan)
        tmp = pd.DataFrame({"price": price, "size": size}).dropna()
        tmp = tmp[tmp["size"] > 0]
        if not tmp.empty:
            series = (tmp["price"] / tmp["size"]).replace([np.inf, -np.inf], np.nan).dropna()

    # Fallback: price only
    if series is None:
        if "price" not in df.columns:
            return 0.0
        series = df["price"].replace([np.inf, -np.inf], np.nan).dropna()

    if series is None or series.empty:
        return 0.0

    mean_v = float(series.mean())
    if mean_v <= 0:
        return 0.0

    coef_var = float(series.std(ddof=0) / mean_v)
    score = min(coef_var * 100.0, 100.0)
    return round(_clip01_100(score), 1)


# -------------------------
# Yield Potential (value proxy from price per sqm) - higher = better
# -------------------------
def yield_potential_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or "size_sqm" not in df.columns or "price" not in df.columns:
        return 0.0

    size = df["size_sqm"].replace([np.inf, -np.inf], np.nan)
    price = df["price"].replace([np.inf, -np.inf], np.nan)

    valid = pd.DataFrame({"price": price, "size": size}).dropna()
    valid = valid[valid["size"] > 0]
    if valid.empty:
        return 0.0

    ppsqm = float((valid["price"] / valid["size"]).median())
    if ppsqm <= 0:
        return 0.0

    denom = (YIELD_PPSQM_WORST - YIELD_PPSQM_BEST)
    if denom <= 0:
        return 0.0

    score = 100.0 - ((ppsqm - YIELD_PPSQM_BEST) / denom) * 100.0
    return round(_clip01_100(score), 1)


# -------------------------
# Momentum (recency proxy) - higher = better
# -------------------------
def momentum_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or "last_seen" not in df.columns:
        return 0.0

    last = pd.to_datetime(df["last_seen"], errors="coerce").dropna()
    if last.empty:
        return 0.0

    try:
        if getattr(last.dt, "tz", None) is not None:
            last = last.dt.tz_convert(None)
    except Exception:
        pass

    now = pd.Timestamp.utcnow().tz_localize(None)

    age_days = (now - last).dt.days
    age_days = age_days[age_days >= 0]
    if age_days.empty:
        return 0.0

    median_age = float(age_days.median())
    score = 100.0 * float(np.exp(-median_age / 30.0))
    return round(_clip01_100(score), 1)


# -------------------------
# Market Depth (proxy: listings count) - higher = better
# Use log-scale to reduce domination by sheer sample size.
# -------------------------
def market_depth_score(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0

    n = float(len(df))
    # log1p: 0->0, 9->~2.3, 99->~4.6, 999->~6.9
    # map to 0-100 with a reasonable cap
    score = min((np.log1p(n) / np.log1p(100.0)) * 100.0, 100.0)  # ~100 listings => 100
    return round(_clip01_100(score), 1)


# -------------------------
# Barzel Score (0-100)
# risk is "bad when high" => invert here only.
# -------------------------
def barzel_score(
    yield_s: float,
    liquidity_s: float,
    risk_s: float,
    momentum_s: float,
    depth_s: float,
) -> float:
    score = (
        0.25 * float(yield_s)
        + 0.25 * float(liquidity_s)
        + 0.20 * (100.0 - float(risk_s))
        + 0.20 * float(momentum_s)
        + 0.10 * float(depth_s)
    )
    return round(_clip01_100(score), 1)
