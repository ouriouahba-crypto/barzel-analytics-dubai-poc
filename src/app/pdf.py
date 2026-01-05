# src/app/pdf.py
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Single source of truth for confidence/gating (align with engine.py)
from src.processing.engine import RECO_MIN_N, confidence_label, can_recommend

# ------------------------------------------------------------
# Demo metadata (used in every memo)
# ------------------------------------------------------------
BRAND = "Barzel Analytics"
DATASET_NAME = "Dubai POC"
DATASET_VERSION = "v0.1"

# Global conventions
RISK_CONVENTION = "Risk Index: higher = more risky (dispersion proxy)."
YIELD_CONVENTION = "Yield Potential: value proxy from median price/sqm (not actual rental yield)."
SCREENING_DISCLAIMER = "Screening tool only — not underwriting, pricing, or investment advice."


# ------------------------------------------------------------
# Small helpers (shared)
# ------------------------------------------------------------
def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_num(x: Any, decimals: int = 1) -> str:
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "—"


def _fmt_int(x: Any) -> str:
    try:
        return f"{int(x)}"
    except Exception:
        return "—"


def _safe_str(x: Any) -> str:
    try:
        s = str(x)
        return s if s.strip() else "—"
    except Exception:
        return "—"


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


def _draw_data_provenance(
    c: canvas.Canvas,
    y: float,
    df_all: pd.DataFrame,
    districts: list[str],
) -> float:
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
        f"Coverage — Marina: {counts.get('Marina', 0)} | Business Bay: {counts.get('Business Bay', 0)} | JVC: {counts.get('JVC', 0)}",
    )
    y -= 14
    c.drawString(60, y, "Refresh cadence: demo cache (TTL varies by page)")
    return y - 8


# ============================================================
# 1) Executive Snapshot memo
# ============================================================
def build_executive_memo_pdf(
    metrics_df: pd.DataFrame,
    df_all: pd.DataFrame,
    districts: Optional[list[str]] = None,
) -> bytes:
    """One-page *analytical* executive summary (no recommendation on the document).

    This memo is meant to be:
    - descriptive
    - defensible
    - compliant-friendly (screening view)
    """
    districts = districts or ["Marina", "Business Bay", "JVC"]

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    _draw_header(c, "Dubai Executive Data Summary")

    # Provenance
    _draw_data_provenance(c, df_all=df_all, districts=districts)

    # Key metrics table (compact)
    y = height - 220
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Key Metrics (Listings dataset)")
    y -= 18

    c.setFont("Helvetica", 9)
    cols = [
        ("District", 40),
        ("Listings", 140),
        ("Weighted AED/sqm", 200),
        ("Median AED/sqm", 310),
        ("Median Days Active", 410),
    ]
    # header
    for label, x in cols:
        c.drawString(x, y, label)
    y -= 12

    # rows
    if metrics_df is None or metrics_df.empty:
        rows = []
    else:
        rows = metrics_df.copy()

    for _, row in (rows.iterrows() if hasattr(rows, "iterrows") else []):
        if y < 120:
            break
        district = _safe_str(row.get("District", "—"))
        listings = _fmt_int(row.get("Listings", None))
        w_ppsqm = _fmt_num(row.get("Weighted Price/sqm (AED)", None), 0)
        med_ppsqm = _fmt_num(row.get("Median Price/sqm (AED)", None), 0)
        med_days = _fmt_num(row.get("Median Days Active", None), 1)

        c.drawString(40, y, district)
        c.drawString(140, y, listings)
        c.drawString(200, y, w_ppsqm)
        c.drawString(310, y, med_ppsqm)
        c.drawString(410, y, med_days)
        y -= 12

    # Limitations
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Notes & Limitations")
    y -= 18
    c.setFont("Helvetica", 10)
    lines = [
        "This dashboard uses listing-level data (not notarized transactions).",
        "Days Active is a listing lifetime proxy: last_seen - first_seen.",
        "No underwriting: this summary is descriptive screening only.",
        "For investment decisions, use verified transactions, rent rolls, fees, and capex assumptions.",
    ]
    for line in lines:
        c.drawString(60, y, line)
        y -= 14

    _footer_disclaimer(c, width, height)
    c.showPage()
    c.save()
    return buffer.getvalue()


def build_compare_memo_pdf(
    metrics_df: pd.DataFrame,
    df_all: pd.DataFrame,
    districts: list[str] | None = None,
) -> bytes:
    """One-page comparison memo (analysis only, no winner/recommendation)."""
    districts = districts or ["Marina", "Business Bay", "JVC"]

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    _draw_header(c, "Dubai Compare Memo")

    _draw_data_provenance(c, df_all=df_all, districts=districts)

    y = height - 220
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Cross-District Comparison (Proxies)")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(60, y, "Primary metrics are descriptive: price/sqm and listing lifetime (Days Active).")

    y -= 22
    c.setFont("Helvetica-Bold", 10)
    cols = [
        ("District", 40),
        ("Listings", 130),
        ("Median Price (AED)", 190),
        ("Median Size (sqm)", 300),
        ("Weighted AED/sqm", 400),
    ]
    for label, x in cols:
        c.drawString(x, y, label)
    y -= 12
    c.setFont("Helvetica", 9)

    if metrics_df is None or metrics_df.empty:
        rows = []
    else:
        rows = metrics_df.copy()

    for _, row in (rows.iterrows() if hasattr(rows, "iterrows") else []):
        if y < 120:
            break
        c.drawString(40, y, _safe_str(row.get("District", "—")))
        c.drawString(130, y, _fmt_int(row.get("Listings", None)))
        c.drawString(190, y, _fmt_num(row.get("Median Price (AED)", None), 0))
        c.drawString(300, y, _fmt_num(row.get("Median Size (sqm)", None), 1))
        c.drawString(400, y, _fmt_num(row.get("Weighted Price/sqm (AED)", None), 0))
        y -= 12

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "How to read")
    y -= 18
    c.setFont("Helvetica", 10)
    for line in [
        "Lower AED/sqm can imply better entry pricing (value proxy), but it may also reflect stock mix.",
        "Median Days Active is a liquidity proxy; lower can indicate faster absorption (subject to listing practices).",
        "Always validate with verified transactions and segment by typology (Studio/1BR/2BR...).",
    ]:
        c.drawString(60, y, line)
        y -= 14

    _footer_disclaimer(c, width, height)
    c.showPage()
    c.save()
    return buffer.getvalue()


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
    c.drawString(40, height - 140, f"Filters: District={_safe_str(district)} | Bedrooms={bedrooms if bedrooms else 'All'}")
    c.drawString(40, height - 155, f"Sample size: n={_fmt_int(n)} | Confidence={_safe_str(conf)}")

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

    if str(conf).lower() == "low":
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
            call_parts.append(
                "high liquidity (proxy)"
                if liq_local >= 70
                else ("medium liquidity (proxy)" if liq_local >= 40 else "low liquidity (proxy)")
            )
        else:
            call_parts.append("unknown liquidity")

        c.drawString(60, y, f"Signal indicates: {', '.join(call_parts)}.")
        y -= 16

    c.setFont("Helvetica", 10)
    c.drawString(60, y, "Micro read is local/descriptive and does not affect the Recommendation engine.")

    y -= 26
    _draw_data_provenance(c, y, df_all, districts)

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

    _draw_header(c, "Dubai Investment Memo")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 140, f"Investor profile: {_safe_str(profile)}")

    n_listings = int(top_row.get("listings", 0)) if top_row is not None else 0
    conf = str(top_row.get("confidence", confidence_label(n_listings))) if top_row is not None else confidence_label(n_listings)
    ok = bool(top_row.get("can_recommend", can_recommend(n_listings))) if top_row is not None else False

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

    c.drawString(60, y, f"#1 Top-ranked: {_safe_str(top_row.get('district'))}")
    y -= 16

    if fallback_row is not None:
        c.drawString(60, y, f"#2 Fallback: {_safe_str(fallback_row.get('district'))}")
        y -= 16

    if aggressive_row is not None:
        c.drawString(60, y, f"#3 Upside option: {_safe_str(aggressive_row.get('district'))} (value proxy / yield-max)")
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
    _draw_data_provenance(c, y, df_all, districts)

    _footer_disclaimer(c)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
