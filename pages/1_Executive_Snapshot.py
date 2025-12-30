# pages/1_Executive_Snapshot.py
import streamlit as st
import pandas as pd
from sqlalchemy import text

from src.app.pdf import build_executive_memo_pdf
from src.app.guardrails import require_non_empty, now_utc_str, district_counts, warn_low_coverage
from src.db.db import get_engine
from src.processing.engine import compute_scores_df, pick_recommendation

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


df = load_data()
require_non_empty(df, "listings")

# Soft quality warnings (don’t block demo)
warn_low_coverage(df, "price", 0.70, "price")
warn_low_coverage(df, "size_sqm", 0.60, "size_sqm")

# Fund language (still supports your old profiles via aliases in engine.py)
profile = st.selectbox(
    "Investor profile",
    ["Capital Preservation", "Core", "Core+", "Opportunistic"],
    index=0,
)

hero(
    "Executive Snapshot",
    "Decision-grade overview (0–100 proxies) for Marina / Business Bay / JVC.",
    pills=[
        pill("B2B Demo", "good"),
        pill("Screening tool", "warn"),
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
- **Risk Index (Dispersion proxy)**: coefficient of variation (based on **price per sqm** when available). Higher = more dispersion (more risk).
- **Momentum (Recency proxy)**: exponential decay based on median listing age.
- **Market Depth (Coverage proxy)**: proxy from listing count (data coverage, not true market size).
- **Barzel Score (0–100)**: weighted blend (risk is inverted inside the score).
        """
    )

scores_df = compute_scores_df(df, districts=DISTRICTS)
if scores_df is None or scores_df.empty:
    st.error("Scores could not be computed. Check numeric fields (price, size_sqm, timestamps).")
    st.stop()

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

top = pick_recommendation(scores_df, profile)

# Decision banner (fund-grade + gated)
if top.get("district") is None:
    st.warning(
        f"No recommendation for **{profile}**: {top.get('reason', 'insufficient data')}"
    )
    st.caption("We still show the ranking for transparency, but we do not output a decision under low coverage.")
    recommended = "—"
else:
    recommended = str(top["district"])
    st.success(
        f"Decision (screening): **{recommended}** ranks #1 for **{profile}** based on current proxies."
    )

st.caption(
    "Interpretation: higher is better for all KPIs except **Risk Index** (higher = more risky). "
    "Barzel Score already accounts for this by inverting risk internally."
)

st.markdown("### KPI Recap (0–100)")
recap_cols = ["district", "barzel", "adoption", "yield", "risk", "momentum", "depth", "listings", "confidence", "can_recommend"]
recap = scores_df[[c for c in recap_cols if c in scores_df.columns]].copy()
st.dataframe(recap.sort_values("barzel", ascending=False), use_container_width=True, hide_index=True)

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
    scores_df=scores_df,
)

safe_profile = profile.lower().replace(" ", "_").replace("-", "_")
safe_reco = recommended.lower().replace(" ", "_").replace("-", "_")

st.download_button(
    label="📄 Download 1-page Executive Memo (PDF)",
    data=pdf_bytes,
    file_name=f"barzel_executive_memo_{safe_reco}_{safe_profile}.pdf",
    mime="application/pdf",
)
