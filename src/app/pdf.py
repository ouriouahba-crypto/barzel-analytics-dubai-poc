# src/app/pdf.py
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Demo metadata (used in every memo)
DATASET_NAME = "Dubai POC"
DATASET_VERSION = "v0.1"
BRAND = "Barzel Analytics"

# Global conventions
RISK_CONVENTION = "Risk Index: higher = more risky (dispersion proxy)."
YIELD_CONVENTION = "Yield Potential: value proxy from median price/sqm (not actual rental yield)."
SCREENING_DISCLAIMER = "Screening tool only — not underwriting, pricing, or investment advice."

# Confidence thresholds (should match engine.py)
RECO_MIN_N = 10
CONF_HIGH = 20
CONF_MED = 10


# ----------------------------
# Small helpers (shared)
# ----------------------------
def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_num(x: Any, decimals: int = 1) -> str:
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "—"


def _safe_slug(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def _confidence_label(n: int) -> str:
    if n >= CONF_HIGH:
        return "High"
    if n >= CONF_MED:
        return "Medium"
    return "Low"


def _can_recommend(n: int) -> bool:
    return n >= RECO_MIN_N


def _district_counts(df_all: pd.DataFrame, districts: list[str]) -> dict[str, int]:
    counts = {d: 0 for d in districts}
    if df_all is None or df_all.empty or "district" not in df_all.columns:
        return counts
    for d in districts:
        counts[d] = int((df_all["district"] == d).sum())
    return counts


def _draw_header(c: canvas.Canvas, title: str) -> None:
    width, height = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 50, f"{BRAND} — {title}")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 70, f"Generated: {_now_utc_str()}")
    c.drawString(40, height - 85, f"Dataset: {DATASET_NAME} {DATASET_VERSION}")
    c.drawString(40, height - 100, SCREENING_DISCLAIMER)
    c.drawString(40, height - 115, f"{RISK_CONVENTION}  |  {YIELD_CONVENTION}")


def _footer_disclaimer(c: canvas.Canvas) -> None:
    c.setFont("Helvetica", 8)
    c.drawString(40, 18, f"{BRAND} • {DATASET_NAME} {DATASET_VERSION} • {SCREENING_DISCLAIMER}")


def _draw_data_provenance(c: canvas.Canvas, y: float, df_all: pd.DataFrame, districts: list[str]) -> float:
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Data Provenance")
    y -= 18
    c.setFont("Helvetica", 10)

    total = len(df_all) if df_all is not None else 0
    counts = _district_counts(df_all, districts)

    c.drawString(60, y, f"Total listings: {total}")
    y -= 14
    c.drawString(
        60,
        y,
        f"Coverage — Marina: {counts.get('Marina',0)} | Business Bay: {counts.get('Business Bay',0)} | JVC: {counts.get('JVC',0)}",
    )
    y -= 14
    c.drawString(60, y, "Refresh cadence: demo cache (TTL varies by page)")
    return y - 8


# ============================================================
# 1) Executive Snapshot memo
# ============================================================
def build_executive_memo_pdf(
    profile: str,
    recommended: str,
    scores: dict[str, dict[str, float | int | str | bool]],
    scores_df: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    _draw_header(c, "Dubai Executive Memo")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 140, f"Investor profile: {profile}")

    # Decision line (gated)
    rec = scores.get(recommended, {}) if recommended and recommended != "—" else {}
    n = int(rec.get("listings", 0)) if rec else 0
    conf = str(rec.get("confidence", _confidence_label(n))) if rec else _confidence_label(n)
    ok = bool(rec.get("can_recommend", _can_recommend(n))) if rec else False

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 165, "Decision")

    c.setFont("Helvetica", 11)
    if recommended == "—" or not ok:
        c.drawString(60, height - 183, f"No recommendation (min {RECO_MIN_N} listings required).")
        c.drawString(60, height - 199, f"Top candidate shown for transparency only.")
    else:
        c.drawString(60, height - 183, f"Top pick: {recommended} for {profile} (screening).")

    c.setFont("Helvetica", 10)
    c.drawString(60, height - 215, f"Confidence: {conf}  |  Sample size (n): {n}")

    # KPI snapshot (if we have a valid recommended)
    y = height - 245
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "KPI Snapshot (0–100)")
    y -= 18
    c.setFont("Helvetica", 11)

    if not rec:
        c.drawString(60, y, "No KPI snapshot available (no recommendation).")
        y -= 16
    else:
        lines = [
            f"Barzel Score: {_fmt_num(rec.get('barzel'))}/100",
            f"Adoption Speed (Liquidity proxy): {_fmt_num(rec.get('adoption'))}",
            f"Yield Potential (Value proxy): {_fmt_num(rec.get('yield'))}",
            f"Risk Index (Dispersion proxy): {_fmt_num(rec.get('risk'))}  (higher = more risky)",
            f"Momentum (Recency proxy): {_fmt_num(rec.get('momentum'))}",
            f"Market Depth (Coverage proxy): {_fmt_num(rec.get('depth'), 0)}   |   Listings: {n}",
        ]
        for line in lines:
            c.drawString(60, y, line)
            y -= 16

    # Thesis (memo-ready)
    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Investment Thesis (Screening)")
    y -= 18
    c.setFont("Helvetica", 11)

    if recommended == "—" or not ok:
        thesis_lines = [
            "Coverage under current dataset is insufficient for a confident district-level decision.",
            "Use this memo to validate the decision framework and request deeper data / scope if interested.",
        ]
    else:
        other = scores_df[scores_df["district"] != recommended].copy() if scores_df is not None else pd.DataFrame()

        best_adoption_other = float(other["adoption"].max()) if len(other) else None
        best_yield_other = float(other["yield"].max()) if len(other) else None
        # risk: lower is safer but we DISPLAY raw; so best "safer" is min
        safest_other = float(other["risk"].min()) if len(other) else None

        rec_adoption = float(rec.get("adoption", 0.0) or 0.0)
        rec_yield = float(rec.get("yield", 0.0) or 0.0)
        rec_risk = float(rec.get("risk", 0.0) or 0.0)

        adoption_text = (
            "liquidity proxy is competitive"
            if best_adoption_other is None
            else ("liquidity proxy leads peers" if rec_adoption >= best_adoption_other else "liquidity proxy is solid vs peers")
        )
        yield_text = (
            "value proxy is attractive"
            if best_yield_other is None
            else ("strongest value proxy (cheaper per sqm)" if rec_yield >= best_yield_other else "balanced value proxy")
        )
        risk_text = (
            "dispersion appears controlled"
            if safest_other is None
            else ("safest dispersion among peers" if rec_risk <= safest_other else "dispersion trade-off is acceptable")
        )

        thesis_lines = [
            f"{recommended} ranks highest for {profile} under current proxies:",
            f"- {yield_text}",
            f"- {adoption_text}",
            f"- {risk_text} (Risk Index displayed raw; inverted inside composite scores)",
        ]

    for line in thesis_lines:
        c.drawString(60, y, line)
        y -= 16

    # Limitations
    y -= 4
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Limitations")
    y -= 18
    c.setFont("Helvetica", 10)
    limitations = [
        "Signals are listings-based (not transactions).",
        "Yield Potential is a value proxy (price/sqm), not rental yield.",
        "Liquidity is proxied via DOM + relative volume share.",
    ]
    for line in limitations:
        c.drawString(60, y, line)
        y -= 14

    _footer_disclaimer(c)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


# ============================================================
# 2) Compare memo
# ============================================================
def build_compare_memo_pdf(
    compare_df: pd.DataFrame,
    recommended: str,
    df_all: pd.DataFrame,
    districts: list[str] | None = None,
) -> bytes:
    districts = districts or ["Marina", "Business Bay", "JVC"]

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    _draw_header(c, "Dubai Compare Memo")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 140, "View: Yield Potential (value proxy) vs Adoption Speed (liquidity proxy)")

    # Decision line
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 165, "Trade-off Pick")
    c.setFont("Helvetica", 11)
    if recommended == "—":
        c.drawString(60, height - 183, f"No recommendation (min {RECO_MIN_N} listings required).")
    else:
        c.drawString(60, height - 183, f"Top candidate (screening): {recommended}")

    y = height - 215
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "KPI Snapshot (0–100)")
    y -= 18

    cols = ["District", "Yield Potential", "Adoption Speed", "Risk Index", "Barzel Score"]
    rows = compare_df[cols].copy() if compare_df is not None else pd.DataFrame(columns=cols)

    c.setFont("Helvetica-Bold", 10)
    header = f"{'District':<16} {'YieldPot':>8} {'Adoption':>9} {'Risk':>7} {'Barzel':>7}"
    c.drawString(60, y, header)
    y -= 14
    c.setFont("Helvetica", 10)

    for _, r in rows.iterrows():
        try:
            line = (
                f"{str(r['District']):<16} "
                f"{float(r['Yield Potential']):>8.1f} "
                f"{float(r['Adoption Speed']):>9.1f} "
                f"{float(r['Risk Index']):>7.1f} "
                f"{float(r['Barzel Score']):>7.1f}"
            )
        except Exception:
            continue
        c.drawString(60, y, line)
        y -= 14

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Interpretation (Screening)")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(60, y, "Upper-right is the target: stronger value proxy with faster absorption (liquidity proxy).")
    y -= 16
    c.drawString(60, y, "Risk Index is displayed raw (higher = more risky) and inverted inside the trade-off score.")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(60, y, "This memo is descriptive for trade-off scanning, not underwriting / pricing.")

    y -= 26
    y = _draw_data_provenance(c, y, df_all, districts)

    _footer_disclaimer(c)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


# ============================================================
# 3) Map & Micro memo
# ============================================================
def build_micro_memo_pdf(
    district: str,
    bedrooms: list,
    n: int,
    conf: str,
    median_price: float,
    median_ppsqm: float,
    median_size: float,
    liq_local: float | None,
    pct_val: float | None,
    iqr_val: float | None,
    df_all: pd.DataFrame,
    districts: list[str] | None = None,
) -> bytes:
    districts = districts or ["Marina", "Business Bay", "JVC"]

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    _draw_header(c, "Dubai Micro Memo")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 140, f"Filters: District={district} | Bedrooms={bedrooms if bedrooms else 'All'}")
    c.drawString(40, height - 155, f"Sample size: n={n} | Confidence={conf}")

    y = height - 185
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Micro Snapshot")
    y -= 18
    c.setFont("Helvetica", 11)

    lines = [
        f"Median Price: {median_price:,.0f} AED",
        f"Median Price / sqm: {median_ppsqm:,.0f} AED/sqm (value proxy)",
        f"Median Size: {median_size:,.0f} sqm",
    ]
    for line in lines:
        c.drawString(60, y, line)
        y -= 16

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Micro Signals (Screening)")
    y -= 18
    c.setFont("Helvetica", 11)

    liq_display = f"{liq_local:.1f}" if liq_local is not None else "— (select single district)"
    prem_display = f"P{pct_val:.0f}" if pct_val is not None else "—"
    iqr_display = f"{iqr_val:,.0f} AED/sqm" if iqr_val is not None else "—"

    sig_lines = [
        f"Liquidity (local Adoption Speed proxy): {liq_display}",
        f"Pricing level percentile (vs global ppsqm): {prem_display}",
        f"Pricing tightness (IQR ppsqm): {iqr_display}  (lower = tighter)",
    ]
    for line in sig_lines:
        c.drawString(60, y, line)
        y -= 16

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Micro Call")
    y -= 18
    c.setFont("Helvetica", 11)

    if conf == "Low":
        c.drawString(60, y, "Low confidence under current filters — descriptive only (avoid strong calls).")
        y -= 16
    else:
        call_parts: list[str] = []
        if pct_val is not None:
            call_parts.append("premium" if pct_val >= 75 else ("mid-market" if pct_val >= 40 else "value"))
        else:
            call_parts.append("unknown pricing")

        if iqr_val is not None:
            call_parts.append("tight" if iqr_val <= 2500 else ("moderate" if iqr_val <= 6000 else "wide"))
        else:
            call_parts.append("unknown dispersion")

        if liq_local is not None:
            call_parts.append("high liquidity (proxy)" if liq_local >= 70 else ("medium liquidity (proxy)" if liq_local >= 40 else "low liquidity (proxy)"))
        else:
            call_parts.append("unknown liquidity")

        c.drawString(60, y, f"Signal indicates: {', '.join(call_parts)}.")
        y -= 16

    c.setFont("Helvetica", 10)
    c.drawString(60, y, "Micro read is local/descriptive and does not affect the Recommendation engine.")

    y -= 26
    y = _draw_data_provenance(c, y, df_all, districts)

    _footer_disclaimer(c)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


# ============================================================
# 4) Recommendation memo
# ============================================================
def build_recommendation_memo_pdf(
    profile: str,
    recommended_district: str,
    top_row: pd.Series,
    fallback_row: pd.Series | None,
    aggressive_row: pd.Series | None,
    df_all: pd.DataFrame,
    districts: list[str] | None = None,
) -> bytes:
    districts = districts or ["Marina", "Business Bay", "JVC"]

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    _draw_header(c, "Dubai Recommendation Memo")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 140, f"Investor profile: {profile}")

    n_listings = int(top_row.get("listings", 0)) if top_row is not None else 0
    conf = str(top_row.get("confidence", _confidence_label(n_listings))) if top_row is not None else _confidence_label(n_listings)
    ok = bool(top_row.get("can_recommend", _can_recommend(n_listings))) if top_row is not None else False

    # Decision
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 165, "Decision")
    c.setFont("Helvetica", 11)

    if recommended_district == "—" or not ok:
        c.drawString(60, height - 183, f"No recommendation (min {RECO_MIN_N} listings required).")
        c.drawString(60, height - 199, "Top-ranked district shown for transparency only.")
    else:
        c.drawString(60, height - 183, f"Allocate exposure to {recommended_district} for a {profile} strategy (screening).")

    c.setFont("Helvetica", 10)
    c.drawString(60, height - 215, f"Confidence: {conf}  |  Sample size (n): {n_listings}")

    # KPI snapshot
    y = height - 245
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "KPI Snapshot (0–100)")
    y -= 18
    c.setFont("Helvetica", 11)

    lines = [
        f"Profile Score: {_fmt_num(top_row.get('profile_score'))}",
        f"Barzel Score: {_fmt_num(top_row.get('barzel'))}/100",
        f"Adoption Speed (Liquidity proxy): {_fmt_num(top_row.get('adoption'))}",
        f"Yield Potential (Value proxy): {_fmt_num(top_row.get('yield'))}",
        f"Risk Index (Dispersion proxy): {_fmt_num(top_row.get('risk'))}  (higher = more risky)",
        f"Momentum (Recency proxy): {_fmt_num(top_row.get('momentum'))}",
        f"Market Depth (Coverage proxy): {_fmt_num(top_row.get('depth'), 0)}   |   Listings: {n_listings}",
    ]
    for line in lines:
        c.drawString(60, y, line)
        y -= 16

    # Shortlist
    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Shortlist")
    y -= 18
    c.setFont("Helvetica", 11)

    c.drawString(60, y, f"#1 Top-ranked: {str(top_row.get('district'))}")
    y -= 16

    if fallback_row is not None:
        c.drawString(60, y, f"#2 Fallback: {str(fallback_row.get('district'))}")
        y -= 16

    if aggressive_row is not None:
        c.drawString(60, y, f"#3 Upside option: {str(aggressive_row.get('district'))} (value proxy / yield-max)")
        y -= 16

    # Limitations
    y -= 4
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Limitations")
    y -= 18
    c.setFont("Helvetica", 10)

    limitations = [
        "Signals are listings-based (not transactions).",
        "Yield Potential is a value proxy (price/sqm), not rental yield.",
        "Liquidity is proxied via DOM + relative volume share.",
        "Composite scores invert risk internally; Risk Index is displayed raw.",
    ]
    for line in limitations:
        c.drawString(60, y, line)
        y -= 14

    y -= 10
    y = _draw_data_provenance(c, y, df_all, districts)

    _footer_disclaimer(c)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
