# src/processing/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd

from src.processing.kpis import (
    adoption_speed_score,
    yield_potential_score,
    risk_index_score,
    momentum_score,
    market_depth_score,
    barzel_score,
)

DEFAULT_DISTRICTS: List[str] = ["Marina", "Business Bay", "JVC"]

# -------------------------
# Confidence / gating
# -------------------------
@dataclass(frozen=True)
class ConfidenceRule:
    min_n: int
    label: str

# Adjust thresholds if you want, but keep them constant across the app + PDFs
CONFIDENCE_RULES: Tuple[ConfidenceRule, ...] = (
    ConfidenceRule(min_n=20, label="High"),
    ConfidenceRule(min_n=10, label="Medium"),
    ConfidenceRule(min_n=0, label="Low"),
)

# Recommendation is blocked under this threshold
RECO_MIN_N = 10


def confidence_label(n: int) -> str:
    for rule in CONFIDENCE_RULES:
        if n >= rule.min_n:
            return rule.label
    return "Low"


def can_recommend(n: int) -> bool:
    return n >= RECO_MIN_N


# -------------------------
# KPI conventions
# -------------------------
def _as_score(x) -> float:
    """Coerce to float score in [0,100] if possible, else 0."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    # Soft clamp to avoid weird KPI explosions
    if v < 0:
        return 0.0
    if v > 100:
        return 100.0
    return v


def _validate_required_columns(df_all: pd.DataFrame) -> None:
    if "district" not in df_all.columns:
        raise ValueError("Missing required column: 'district'")


# -------------------------
# Core scoring
# -------------------------
def compute_scores_for_district(df_all: pd.DataFrame, district: str) -> Dict:
    """
    Compute the full KPI set for one district.
    Returns a dict with consistent keys used everywhere in the app.

    Convention (IMPORTANT):
    - All KPIs are scores from 0 to 100 where HIGHER = BETTER
      EXCEPT risk, where HIGHER = MORE RISKY (worse).
    - Any profile scoring that wants "low risk is good" must invert risk centrally.
    """
    _validate_required_columns(df_all)

    ddf = df_all[df_all["district"] == district].copy()
    n = int(len(ddf))

    liq = _as_score(adoption_speed_score(ddf, df_all))   # higher = better
    yld = _as_score(yield_potential_score(ddf))          # higher = better
    risk = _as_score(risk_index_score(ddf))              # higher = MORE risky (worse)
    mom = _as_score(momentum_score(ddf))                 # higher = better
    depth = _as_score(market_depth_score(ddf))           # higher = better

    total = _as_score(barzel_score(yld, liq, risk, mom, depth))

    conf = confidence_label(n)
    recommendable = can_recommend(n)

    return {
        "district": district,
        "barzel": round(float(total), 1),
        "adoption": round(float(liq), 1),
        "yield": round(float(yld), 1),
        "risk": round(float(risk), 1),
        "momentum": round(float(mom), 1),
        "depth": round(float(depth), 1),
        "listings": n,
        "confidence": conf,
        "can_recommend": bool(recommendable),
    }


def compute_scores_df(df_all: pd.DataFrame, districts: List[str] | None = None) -> pd.DataFrame:
    """
    Single source of truth for KPI scores across the whole app.
    Returns:
    district | barzel | adoption | yield | risk | momentum | depth | listings | confidence | can_recommend
    """
    if districts is None:
        districts = DEFAULT_DISTRICTS

    rows = [compute_scores_for_district(df_all, d) for d in districts]
    df = pd.DataFrame(rows)

    # Stable column order
    cols = [
        "district",
        "barzel",
        "adoption",
        "yield",
        "risk",
        "momentum",
        "depth",
        "listings",
        "confidence",
        "can_recommend",
    ]
    return df[cols]


# -------------------------
# Investor profiles (fund language)
# -------------------------
PROFILE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Lowest volatility / safest bias
    "Capital Preservation": {"adoption": 0.35, "risk": 0.35, "yield": 0.10, "depth": 0.10, "momentum": 0.10},
    # Typical long-term allocation
    "Core": {"adoption": 0.25, "risk": 0.25, "yield": 0.20, "depth": 0.15, "momentum": 0.15},
    # Slightly more return-seeking
    "Core+": {"adoption": 0.25, "risk": 0.20, "yield": 0.25, "depth": 0.15, "momentum": 0.15},
    # Return-max / higher risk tolerance
    "Opportunistic": {"yield": 0.40, "adoption": 0.25, "risk": 0.15, "depth": 0.10, "momentum": 0.10},
}

# Backward-compatible aliases so your existing UI doesn't break
PROFILE_ALIASES: Dict[str, str] = {
    "Conservative": "Capital Preservation",
    "Balanced": "Core",
    "Yield-oriented": "Opportunistic",
}


def _normalize_profile(profile: str) -> str:
    return PROFILE_ALIASES.get(profile, profile)


def rank_for_profile(scores_df: pd.DataFrame, profile: str) -> pd.DataFrame:
    """
    Compute a profile_score and return a sorted dataframe.

    Risk handling:
    - scores_df['risk'] is "more is worse"
    - For profile_score we invert it: (100 - risk) means safer -> higher score
    """
    p = _normalize_profile(profile)
    if p not in PROFILE_WEIGHTS:
        raise ValueError(f"Unknown profile '{profile}'. Allowed: {list(PROFILE_WEIGHTS.keys())} (+ aliases)")

    w = PROFILE_WEIGHTS[p]

    df = scores_df.copy()

    # If you accidentally pass a df without these columns, fail fast
    required = {"yield", "adoption", "depth", "momentum", "risk", "listings", "can_recommend"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rank_for_profile: missing columns: {sorted(missing)}")

    df["profile_score"] = (
        w.get("yield", 0) * df["yield"]
        + w.get("adoption", 0) * df["adoption"]
        + w.get("depth", 0) * df["depth"]
        + w.get("momentum", 0) * df["momentum"]
        + w.get("risk", 0) * (100 - df["risk"])  # invert risk centrally
    ).round(1)

    # Optional: push non-recommendable districts down (still visible, just ranked last)
    # This avoids "best is n=6" situations.
    df["_reco_penalty"] = df["can_recommend"].apply(lambda x: 0 if x else 1000)
    df = df.sort_values(["_reco_penalty", "profile_score"], ascending=[True, False]).drop(columns=["_reco_penalty"])

    return df.reset_index(drop=True)


def pick_recommendation(scores_df: pd.DataFrame, profile: str) -> Dict:
    ranked = rank_for_profile(scores_df, profile)

    # First try: recommendable only
    if "can_recommend" in ranked.columns:
        ok = ranked[ranked["can_recommend"] == True]
        if not ok.empty:
            top = ok.iloc[0].to_dict()
            top["profile"] = _normalize_profile(profile)
            return top

    # Else: no reco payload with best candidate shown
    top_candidate = ranked.iloc[0].to_dict() if not ranked.empty else {}
    return {
        "district": None,
        "reason": f"Insufficient data to recommend (min {RECO_MIN_N} listings required).",
        "profile": _normalize_profile(profile),
        "top_candidate": top_candidate,
    }
