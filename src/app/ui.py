# src/app/ui.py
from __future__ import annotations

import html
import streamlit as st


def inject_base_ui() -> None:
    """Global CSS + minor Streamlit layout tweaks (clean light theme)."""

    st.markdown(
        """
<style>
:root{
  --bg: #F6F8FC;
  --card: #FFFFFF;
  --card2: #F0F5FF;
  --border: rgba(11,18,32,0.10);
  --text: rgba(11,18,32,0.92);
  --muted: rgba(11,18,32,0.60);
  --muted2: rgba(11,18,32,0.45);
  --good: #16A34A;
  --warn: #F59E0B;
  --bad:  #DC2626;
  --accent: #0EA5A4;
  --shadow: 0 12px 28px rgba(15, 23, 42, 0.10);
}

section.main > div { padding-top: 1.0rem; }
.block-container { max-width: 1380px; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Hero */
.ba-hero {
  background: linear-gradient(135deg, rgba(14,165,164,0.16), rgba(59,130,246,0.10));
  border: 1px solid rgba(14,165,164,0.25);
  border-radius: 20px;
  padding: 16px 18px 14px 18px;
  box-shadow: var(--shadow);
}
.ba-title { font-size: 1.40rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; color: var(--text); }
.ba-subtitle { margin-top: 0.25rem; color: var(--muted) !important; font-size: 0.95rem; }

/* Metric cards */
.ba-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-top: 12px;
}
@media (max-width: 1050px) {
  .ba-metrics { grid-template-columns: repeat(2, 1fr); }
}
.ba-metric {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px 14px 10px 14px;
  box-shadow: 0 8px 18px rgba(15,23,42,0.06);
}
.ba-metric .k { color: var(--muted) !important; font-size: 0.82rem; margin-bottom: 4px; }
.ba-metric .v { font-size: 1.12rem; font-weight: 800; line-height: 1.2; color: var(--text); }
.ba-metric .h { color: var(--muted2) !important; font-size: 0.78rem; margin-top: 2px; }

/* Pills */
.ba-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  border: 1px solid var(--border);
  background: rgba(2,6,23,0.02);
  color: var(--text);
}
.ba-pill.good { border-color: rgba(22,163,74,0.35); background: rgba(22,163,74,0.08); color: rgba(22,163,74,0.92); }
.ba-pill.warn { border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.10); color: rgba(180,83,9,0.92); }
.ba-pill.bad  { border-color: rgba(220,38,38,0.35); background: rgba(220,38,38,0.08); color: rgba(220,38,38,0.92); }

/* Inline signals */
.ba-pos { color: var(--good); font-weight: 800; }
.ba-neg { color: var(--bad);  font-weight: 800; }
.ba-neu { color: rgba(59,130,246,0.95); font-weight: 800; }

/* Dataframes */
[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: 0 8px 18px rgba(15,23,42,0.05);
}

/* Download button */
.stDownloadButton button {
  border-radius: 14px !important;
  padding: 0.65rem 1.0rem !important;
  border: 1px solid rgba(14,165,164,0.40) !important;
}

hr { border-top: 1px solid var(--border); }

/* Plotly charts as cards */
[data-testid="stPlotlyChart"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 10px;
  box-shadow: 0 10px 22px rgba(15,23,42,0.08);
}

h1, h2, h3 { letter-spacing: -0.02em; }
</style>
        """,
        unsafe_allow_html=True,
    )


def pill(label: str, tone: str = "good") -> str:
    tone = tone if tone in ("good", "warn", "bad") else "good"
    return f'<span class="ba-pill {tone}">{html.escape(label)}</span>'


def pos(text: str) -> str:
    return f'<span class="ba-pos">{html.escape(text)}</span>'


def neg(text: str) -> str:
    return f'<span class="ba-neg">{html.escape(text)}</span>'


def neu(text: str) -> str:
    return f'<span class="ba-neu">{html.escape(text)}</span>'


def delta_badge(delta: float, *, higher_is_better: bool = True, unit: str = "%") -> str:
    """Return a green/red inline delta string (for metric cards).

    Example: delta_badge(-12.4, higher_is_better=False) -> green (faster)
    """
    if delta is None or not isinstance(delta, (int, float)):
        return neu("—")

    good = delta >= 0 if higher_is_better else delta <= 0
    arrow = "▲" if delta >= 0 else "▼"

    if unit == "%":
        txt = f"{arrow} {delta:+.1f}%"
    else:
        txt = f"{arrow} {delta:+.0f}{unit}"

    return pos(txt) if good else neg(txt)


def hero(title: str, subtitle: str, pills: list[str] | None = None) -> None:
    pills_html = ""
    if pills:
        pills_html = " ".join(pills)
        pills_html = f'<div style="margin-top:10px;">{pills_html}</div>'

    st.markdown(
        f"""
<div class="ba-hero">
  <div class="ba-title">{html.escape(title)}</div>
  <div class="ba-subtitle">{html.escape(subtitle)}</div>
  {pills_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def metric_grid(items: list[tuple[str, str, str | None]]) -> None:
    blocks: list[str] = []
    for k, v, h in items:
        h_html = f'<div class="h">{h}</div>' if h else ""
        blocks.append(
            f"""
<div class="ba-metric">
  <div class="k">{html.escape(k)}</div>
  <div class="v">{v}</div>
  {h_html}
</div>
            """.strip()
        )
    st.markdown(f'<div class="ba-metrics">{"".join(blocks)}</div>', unsafe_allow_html=True)
