# src/processing/kpis.py
from __future__ import annotations

import numpy as np
import pandas as pd

SCORE_MIN = 0.0
SCORE_MAX = 100.0


def _clip01_100(x: float) -> float:
    return float(np.clip(float(x), SCORE_MIN, SCORE_MAX))


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _safe_ppsqm(df: pd.DataFrame) -> pd.Series:
    """Return price per sqm series (cleaned), empty if not computable."""
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if "price" not in df.columns or "size_sqm" not in df.columns:
        return pd.Series(dtype="float64")

    price = pd.to_numeric(df["price"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    size = pd.to_numeric(df["size_sqm"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    tmp = pd.DataFrame({"price": price, "size": size}).dropna()
    tmp = tmp[tmp["size"] > 0]
    if tmp.empty:
        return pd.Series(dtype="float64")

    return (tmp["price"] / tmp["size"]).replace([np.inf, -np.inf], np.nan).dropna()


def _district_medians(all_df: pd.DataFrame, col: str) -> dict:
    """Return dict: district -> median(col), dropping NaNs."""
    if all_df is None or all_df.empty or "district" not in all_df.columns:
        return {}
    if col not in all_df.columns:
        return {}
    s = pd.to_numeric(all_df[col], errors="coerce")
    tmp = all_df.copy()
    tmp[col] = s
    grp = tmp.groupby("district")[col].median(numeric_only=True)
    return {k: float(v) for k, v in grp.dropna().to_dict().items()}


def _minmax_from_map(m: dict) -> tuple[float | None, float | None]:
    if not m:
        return None, None
    vals = [v for v in m.values() if v is not None and np.isfinite(v)]
    if not vals:
        return None, None
    return float(min(vals)), float(max(vals))


def _score_linear(value: float, vmin: float, vmax: float, higher_is_better: bool) -> float:
    """Map value to 0-100 linearly. Handles vmin==vmax safely."""
    if value is None or not np.isfinite(value):
        return 0.0
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0
    if vmax <= vmin:
        return 50.0

    x = (value - vmin) / (vmax - vmin)
    x = float(np.clip(x, 0.0, 1.0))
    score = 100.0 * x if higher_is_better else 100.0 * (1.0 - x)
    return _clip01_100(score)


def adoption_speed_score(df: pd.DataFrame, all_df: pd.DataFrame | None = None) -> float:
    if df is None or df.empty:
        return 0.0

    if all_df is None or all_df.empty or "district" not in all_df.columns:
        if "first_seen" not in df.columns or "last_seen" not in df.columns:
            return 0.0
        d = df.copy()
        d["first_seen"] = _safe_to_datetime(d["first_seen"])
        d["last_seen"] = _safe_to_datetime(d["last_seen"])
        dom = (d["last_seen"] - d["first_seen"]).dt.days
        dom = dom.dropna()
        if dom.empty:
            return 0.0
        med_dom = float(max(dom.median(), 0.0))
        return round(_clip01_100(100.0 - min(med_dom, 100.0)), 1)

    a = all_df.copy()
    a["first_seen"] = _safe_to_datetime(a["first_seen"]) if "first_seen" in a.columns else pd.NaT
    a["last_seen"] = _safe_to_datetime(a["last_seen"]) if "last_seen" in a.columns else pd.NaT

    dom_all = (a["last_seen"] - a["first_seen"]).dt.days
    a["_dom"] = pd.to_numeric(dom_all, errors="coerce")
    a["_dom"] = a["_dom"].where(a["_dom"] >= 0)

    dom_map = _district_medians(a, "_dom")
    vmin, vmax = _minmax_from_map(dom_map)

    district = str(df["district"].iloc[0]) if "district" in df.columns and not df.empty else None
    med_dom = dom_map.get(district, None)

    score = _score_linear(med_dom, vmin, vmax, higher_is_better=False)
    return round(score, 1)


def yield_potential_score(df: pd.DataFrame, all_df: pd.DataFrame | None = None) -> float:
    if df is None or df.empty:
        return 0.0

    if all_df is None or all_df.empty or "district" not in all_df.columns:
        ppsqm = _safe_ppsqm(df)
        if ppsqm.empty:
            return 0.0
        med = float(ppsqm.median())
        return round(_clip01_100(100.0 * np.exp(-med / 40000.0)), 1)

    a = all_df.copy()
    a_price = pd.to_numeric(a["price"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    a_size = pd.to_numeric(a["size_sqm"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = a_price.notna() & a_size.notna() & (a_size > 0)
    a["_ppsqm"] = np.where(mask, a_price / a_size, np.nan)

    ppsqm_map = _district_medians(a, "_ppsqm")
    vmin, vmax = _minmax_from_map(ppsqm_map)

    district = str(df["district"].iloc[0]) if "district" in df.columns and not df.empty else None
    med_ppsqm = ppsqm_map.get(district, None)

    score = _score_linear(med_ppsqm, vmin, vmax, higher_is_better=False)
    return round(score, 1)


def risk_index_score(df: pd.DataFrame, all_df: pd.DataFrame | None = None) -> float:
    if df is None or df.empty:
        return 0.0

    if all_df is None or all_df.empty or "district" not in all_df.columns:
        ppsqm = _safe_ppsqm(df)
        if ppsqm.empty:
            price = pd.to_numeric(df.get("price", pd.Series(dtype="float64")), errors="coerce").dropna()
            if price.empty:
                return 0.0
            mean_v = float(price.mean())
            if mean_v <= 0:
                return 0.0
            cv = float(price.std(ddof=0) / mean_v)
            return round(_clip01_100(cv * 100.0), 1)

        mean_v = float(ppsqm.mean())
        if mean_v <= 0:
            return 0.0
        cv = float(ppsqm.std(ddof=0) / mean_v)
        return round(_clip01_100(cv * 100.0), 1)

    a = all_df.copy()
    a["price"] = pd.to_numeric(a["price"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    a["size_sqm"] = pd.to_numeric(a["size_sqm"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = a["price"].notna() & a["size_sqm"].notna() & (a["size_sqm"] > 0)
    a["_ppsqm"] = np.where(mask, a["price"] / a["size_sqm"], np.nan)

    disp_map = {}
    ticket_map = {}
    for dist, g in a.groupby("district"):
        p = pd.to_numeric(g["_ppsqm"], errors="coerce").dropna()
        pr = pd.to_numeric(g["price"], errors="coerce").dropna()

        if len(p) >= 3 and float(p.mean()) > 0:
            disp_map[str(dist)] = float(p.std(ddof=0) / float(p.mean()))
        if not pr.empty:
            ticket_map[str(dist)] = float(pr.median())

    disp_min, disp_max = _minmax_from_map(disp_map)
    ticket_min, ticket_max = _minmax_from_map(ticket_map)

    district = str(df["district"].iloc[0]) if "district" in df.columns and not df.empty else None
    d_disp = disp_map.get(district, None)
    d_ticket = ticket_map.get(district, None)

    disp_s = _score_linear(d_disp, disp_min, disp_max, higher_is_better=True)
    ticket_s = _score_linear(d_ticket, ticket_min, ticket_max, higher_is_better=True)

    risk = 0.65 * disp_s + 0.35 * ticket_s
    return round(_clip01_100(risk), 1)


def market_depth_score(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    n = float(len(df))
    score = 100.0 * (1.0 - np.exp(-n / 12.0))
    return round(_clip01_100(score), 1)


def momentum_score(df: pd.DataFrame, all_df: pd.DataFrame | None = None) -> float:
    if df is None or df.empty:
        return 0.0
    if "first_seen" not in df.columns or "price" not in df.columns or "size_sqm" not in df.columns:
        return 0.0

    def _momentum_ratio(g: pd.DataFrame) -> float | None:
        g = g.copy()
        g["first_seen"] = _safe_to_datetime(g["first_seen"])
        g["price"] = pd.to_numeric(g["price"], errors="coerce")
        g["size_sqm"] = pd.to_numeric(g["size_sqm"], errors="coerce")
        g = g.dropna(subset=["first_seen", "price", "size_sqm"])
        g = g[g["size_sqm"] > 0]
        if len(g) < 6:
            return None

        g["_ppsqm"] = g["price"] / g["size_sqm"]
        split = g["first_seen"].median()

        old = g[g["first_seen"] < split]["_ppsqm"].dropna()
        new = g[g["first_seen"] >= split]["_ppsqm"].dropna()
        if old.empty or new.empty:
            return None

        ratio = float(new.median() / old.median()) if float(old.median()) > 0 else None
        return ratio

    if all_df is None or all_df.empty or "district" not in all_df.columns:
        r = _momentum_ratio(df)
        if r is None or not np.isfinite(r):
            return 0.0
        score = 50.0 + (r - 1.0) * 200.0
        return round(_clip01_100(score), 1)

    ratios = {}
    for dist, g in all_df.groupby("district"):
        r = _momentum_ratio(g)
        if r is not None and np.isfinite(r):
            ratios[str(dist)] = float(r)

    rmin, rmax = _minmax_from_map(ratios)
    district = str(df["district"].iloc[0]) if "district" in df.columns and not df.empty else None
    r = ratios.get(district, None)

    score = _score_linear(r, rmin, rmax, higher_is_better=True)
    return round(score, 1)


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
