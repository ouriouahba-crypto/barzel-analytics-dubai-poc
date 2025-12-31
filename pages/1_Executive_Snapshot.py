# pages/1_Executive_Snapshot.py
import streamlit as st
import pandas as pd
from sqlalchemy import text

from src.app.pdf import build_executive_memo_pdf
from src.app.guardrails import require_non_empty, now_utc_str, district_counts, warn_low_coverage
from src.db.db import get_engine
from src.processing.engine import compute_scores_df, pick_recommendation, rank_for_profile

from src.app.ui import inject_base_ui, hero, metric_grid, pill

inject_base_ui()

DISTRICTS = ["Marina", "Business Bay", "JVC"]


@st.cache_data(ttl=120)
def load_data() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM listings"), conn)
    return df


def fmt_score(x) -> str:
    try:
        return f"{float(x):.1f}"
    except Exception:
        return "—"


def delta_str(a: float, b: float) -> str:
    try:
        d = float(a) - float(b)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}"
    except Exception:
        return "—"


df = load_data()
require_non_empty(df, "listings")

# Soft warnings (don’t block demo)
warn_low_coverage(df, "price", 0.70, "price")
warn_low_coverage(df, "size_sqm", 0.60, "size_sqm")

profile = st.selectbox(
    "Investor profile",
    ["Capital Preservation", "Core", "Core+", "Opportunistic"],
    index=0,
)

hero(
    "Executive Snapshot",
    "Memo-ready screening decision (0–100 proxies) for Marina / Business Bay / JVC.",
    pills=[
        pill("B2B Demo", "good"),
        pill("Screening, not underwriting", "warn"),
        pill(f"Refresh: {now_utc_str()}", "good"),
    ],
)

with st.expander("Methodology (how scores are built)", expanded=False):
    st.markdown(
        """
**Conventions**
- Scores are **0–100**.
- Higher is better for all KPIs **except** **Risk Index** (higher = more risky).
- This is a **screening tool** (proxy-based), not underwriting.

**KPIs**
- **Adoption Speed (Liquidity proxy)**: blend of median Days-on-Market + relative listing share.
- **Yield Potential (Value proxy)**: derived from **median price per sqm** (lower AED/sqm ⇒ higher score). *Not actual rental yield.*
- **Risk Index (Dispersion proxy)**: dispersion in price per sqm (plus ticket size) — higher = more risky.
- **Momentum (Recency proxy)**: recent-vs-older pricing ratio within district.
- **Market Depth (Coverage proxy)**: saturating proxy from listing count (coverage, not true market size).
- **Barzel Score (0–100)**: weighted blend (risk inverted inside the score).
        """
    )

scores_df = compute_scores_df(df, districts=DISTRICTS)
if scores_df is None or scores_df.empty:
    st.error("Scores could not be computed. Check numeric fields (price, size_sqm, timestamps).")
    st.stop()

ranked = rank_for_profile(scores_df, profile)
reco = pick_recommendation(scores_df, profile)

top = ranked.iloc[0]
runner = ranked.iloc[1] if len(ranked) > 1 else None

# Build dict for PDF + cards
scores: dict[str, dict[str, float | int | str | bool]] = {}
for _, row in scores_df.iterrows():
    d = str(row["district"])
    scores[d] = {
        "barzel": float(row["barzel"]),
        "adoption": float(row["adoption"]),
        "yield": float(row["yield"]),
        "risk": float(row["risk"]),
        "momentum": float(row.get("momentum", 0.0)),
        "depth": float(row["depth"]),
        "listings": int(row["listings"]),
        "confidence": str(row.get("confidence", "—")),
        "can_recommend": bool(row.get("can_recommend", True)),
    }

# -------------------------
# Decision banner (gated)
# -------------------------
if reco.get("district") is None:
    cand = reco.get("top_candidate", {})
    st.warning(f"No recommendation for **{profile}**: {reco.get('reason', 'insufficient data')}")
    st.caption(
        f"Top-ranked (not recommendable): **{cand.get('district','—')}** · "
        f"n={cand.get('listings','—')} · confidence={cand.get('confidence','—')}"
    )
    recommended = "—"
else:
    recommended = str(reco["district"])
    st.success(
        f"Decision (screening): **{recommended}** ranks #1 for **{profile}** under current proxies."
    )
    st.caption(f"Confidence: **{reco.get('confidence','—')}** (n={reco.get('listings','—')} listings)")

st.caption(
    "Interpretation: higher is better for all KPIs except **Risk Index** (higher = more risky). "
    "Barzel Score and Profile Score invert risk internally."
)

# -------------------------
# Memo-ready thesis
# -------------------------
st.divider()
st.subheader("Investment Thesis (screening)")

if runner is None or len(ranked) < 2:
    st.write("Not enough alternatives to generate a robust trade-off thesis.")
else:
    # Pick two primary drivers by profile (memo-grade)
    if profile == "Capital Preservation":
        drivers = [("risk", "Risk (safer is better)"), ("adoption", "Liquidity proxy")]
        tradeoff = ("yield", "Value proxy (price/sqm)")
    elif profile == "Core":
        drivers = [("barzel", "Composite strength"), ("adoption", "Liquidity proxy")]
        tradeoff = ("risk", "Risk (dispersion)")
    elif profile == "Core+":
        drivers = [("yield", "Value proxy (price/sqm)"), ("adoption", "Liquidity proxy")]
        tradeoff = ("risk", "Risk (dispersion)")
    else:  # Opportunistic
        drivers = [("yield", "Value proxy (price/sqm)"), ("momentum", "Momentum (recency proxy)")]
        tradeoff = ("risk", "Risk (dispersion)")

    def _val(row, key):
        try:
            return float(row.get(key, 0.0))
        except Exception:
            return 0.0

    st.markdown(f"**#1 Pick:** {top['district']}  \n**#2 Comparator:** {runner['district']}")
    st.markdown("**Key drivers**")
    for k, label in drivers:
        a = _val(top, k)
        b = _val(runner, k)
        if k == "risk":
            st.write(
                f"• **{label}**: {top['district']} = **{fmt_score(a)}** vs {runner['district']} = **{fmt_score(b)}** "
                f"(Δ {delta_str(a, b)}; lower risk is better)"
            )
        else:
            st.write(
                f"• **{label}**: {top['district']} = **{fmt_score(a)}** vs {runner['district']} = **{fmt_score(b)}** "
                f"(Δ {delta_str(a, b)})"
            )

    tk, tlabel = tradeoff
    a = _val(top, tk)
    b = _val(runner, tk)
    st.markdown("**Main trade-off**")
    if tk == "risk":
        st.write(
            f"• **{tlabel}**: {top['district']} = **{fmt_score(a)}** vs {runner['district']} = **{fmt_score(b)}** "
            f"(Δ {delta_str(a, b)}; higher = more risky)"
        )
    else:
        st.write(
            f"• **{tlabel}**: {top['district']} = **{fmt_score(a)}** vs {runner['district']} = **{fmt_score(b)}** "
            f"(Δ {delta_str(a, b)})"
        )

st.caption("This thesis is proxy-based and intended for screening only (not underwriting).")

# -------------------------
# KPI recap + cards
# -------------------------
st.divider()
st.markdown("### KPI Recap (0–100)")
recap_cols = ["district", "barzel", "adoption", "yield", "risk", "momentum", "depth", "listings", "confidence", "can_recommend", "profile_score"]
recap = ranked[[c for c in recap_cols if c in ranked.columns]].copy()
st.dataframe(recap, use_container_width=True, hide_index=True)

st.markdown("### District Cards")
cols = st.columns(3)
for i, d in enumerate(DISTRICTS):
    with cols[i]:
        st.subheader(d)
        conf = scores[d].get("confidence", "—")
        n = scores[d]["listings"]
        st.caption(f"{n} listings · Confidence: {conf} · Depth {float(scores[d]['depth']):.0f}")
        metric_grid(
            [
                ("Barzel Score", fmt_score(scores[d]["barzel"]), "0–100 (higher is better)"),
                ("Adoption Speed", fmt_score(scores[d]["adoption"]), "Liquidity proxy"),
                ("Yield Potential", fmt_score(scores[d]["yield"]), "Value proxy (price/sqm)"),
                ("Risk Index", fmt_score(scores[d]["risk"]), "Higher = more dispersion (more risk)"),
            ]
        )

st.caption("Risk Index is a dispersion proxy: higher = more price variability (higher risk).")

# -------------------------
# Data provenance + Export
# -------------------------
st.divider()
st.markdown("### Data provenance")
counts = district_counts(df, DISTRICTS)
st.caption(
    f"Data refreshed: **{now_utc_str()}** · Total listings: **{len(df)}** · "
    f"Marina: **{counts['Marina']}** | Business Bay: **{counts['Business Bay']}** | JVC: **{counts['JVC']}**"
)

st.divider()
st.markdown("### Export")

pdf_bytes = build_executive_memo_pdf(
    profile=profile,
    recommended=recommended,
    scores=scores,
    scores_df=ranked,  # send ranked so PDF can compute runner deltas if needed
)

safe_profile = profile.lower().replace(" ", "_").replace("-", "_")
safe_reco = recommended.lower().replace(" ", "_").replace("-", "_")

st.download_button(
    label="📄 Download 1-page Executive Memo (PDF)",
    data=pdf_bytes,
    file_name=f"barzel_executive_memo_{safe_reco}_{safe_profile}.pdf",
    mime="application/pdf",
)
