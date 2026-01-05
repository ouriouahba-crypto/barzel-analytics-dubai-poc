# src/app/ui.py
from __future__ import annotations

import streamlit as st


def inject_base_ui() -> None:
    """Global CSS + minor Streamlit layout tweaks (premium look)."""
    st.markdown(
        """
<style>
:root{
  --bg: #0b1220;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --muted2: rgba(255,255,255,0.50);
  --good: #28d17c;
  --warn: #f5c542;
  --bad: #ff5c5c;
  --accent: #7aa2ff;
}

section.main > div { padding-top: 1.2rem; }
.block-container { max-width: 1380px; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.ba-hero {
  background: linear-gradient(135deg, rgba(122,162,255,0.25), rgba(40,209,124,0.14));
  border: 1px solid rgba(122,162,255,0.25);
  border-radius: 18px;
  padding: 16px 16px 14px 16px;
  box-shadow: 0 12px 28px rgba(0,0,0,0.28);
}

.ba-title { font-size: 1.35rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; }
.ba-subtitle { margin-top: 0.25rem; color: rgba(255,255,255,0.65) !important; font-size: 0.92rem; }

.ba-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin-top: 10px;
}
@media (max-width: 1050px) {
  .ba-metrics { grid-template-columns: repeat(2, 1fr); }
}
.ba-metric {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 12px 10px 12px;
}
.ba-metric .k { color: rgba(255,255,255,0.65) !important; font-size: 0.82rem; margin-bottom: 4px; }
.ba-metric .v { font-size: 1.10rem; font-weight: 700; line-height: 1.2; }
.ba-metric .h { color: rgba(255,255,255,0.50) !important; font-size: 0.78rem; margin-top: 2px; }

.ba-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.92);
}
.ba-pill.good { border-color: rgba(40,209,124,0.45); background: rgba(40,209,124,0.10); }
.ba-pill.warn { border-color: rgba(245,197,66,0.45); background: rgba(245,197,66,0.10); }
.ba-pill.bad  { border-color: rgba(255,92,92,0.45); background: rgba(255,92,92,0.10); }

[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.10);
}

.stDownloadButton button {
  border-radius: 14px !important;
  padding: 0.65rem 1.0rem !important;
  border: 1px solid rgba(122,162,255,0.35) !important;
}

hr { border-top: 1px solid rgba(255,255,255,0.10); }

/* Make Plotly charts feel like premium cards */
[data-testid="stPlotlyChart"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 10px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.22);
}

h1, h2, h3 {
  letter-spacing: -0.02em;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def pill(label: str, tone: str = "good") -> str:
    tone = tone if tone in ("good", "warn", "bad") else "good"
    return f'<span class="ba-pill {tone}">{label}</span>'


def hero(title: str, subtitle: str, pills: list[str] | None = None) -> None:
    pills_html = ""
    if pills:
        pills_html = " ".join(pills)
        pills_html = f'<div style="margin-top:8px;">{pills_html}</div>'

    st.markdown(
        f"""
<div class="ba-hero">
  <div class="ba-title">{title}</div>
  <div class="ba-subtitle">{subtitle}</div>
  {pills_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def metric_grid(items: list[tuple[str, str, str | None]]) -> None:
    blocks = []
    for k, v, h in items:
        h_html = f'<div class="h">{h}</div>' if h else ""
        blocks.append(
            f"""
<div class="ba-metric">
  <div class="k">{k}</div>
  <div class="v">{v}</div>
  {h_html}
</div>
            """.strip()
        )
    st.markdown(f'<div class="ba-metrics">{"".join(blocks)}</div>', unsafe_allow_html=True)
