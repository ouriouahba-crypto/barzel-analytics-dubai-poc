# src/app/formatting.py
from __future__ import annotations

def fmt_score(x) -> str:
    """Score 0-100 with 1 decimal."""
    try:
        return f"{float(x):.1f}"
    except Exception:
        return "—"

def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"

def fmt_aed(x) -> str:
    try:
        return f"{float(x):,.0f} AED"
    except Exception:
        return "—"

def fmt_aed_per_sqm(x) -> str:
    try:
        return f"{float(x):,.0f} AED/sqm"
    except Exception:
        return "—"
