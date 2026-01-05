# pages/4_Recommendation.py
import streamlit as st
import pandas as pd
from sqlalchemy import text

from src.app.pdf import build_recommendation_memo_pdf
from src.app.guardrails import require_non_empty, now_utc_str, district_counts, warn_low_coverage
from src.db.db import get_engine
from src.processing.engine import compute_scores_df, rank_for_profile, pick_recommendation

from src.app.ui import inject_base_ui, hero, metric_grid, pill

inject_base_ui()

DISTRICTS = ["Marina", "Business Bay", "JVC"]


@st.cache_data(ttl=120)
def load_data() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM listings"), conn)

    for col in ["price", "size_sqm", "bedrooms", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


df_all = load_data()
require_non_empty(df_all, "listings")

warn_low_coverage(df_all, "size_sqm", min_ratio=0.80, label="Size (sqm)")
warn_low_coverage(df_all, "price", min_ratio=0.95, label="Price")

counts = district_counts(df_all, DISTRICTS)

profile = st.selectbox(
    "Investor profile (used inside the PDF memo)",
    ["Capital Preservation", "Core", "Core+", "Opportunistic"],
    index=1,
)

hero(
    "PDF Memo Builder",
    "Generate a committee-style memo as a PDF. The dashboard UI stays analysis-only (no on-screen conclusion).",
    pills=[
        pill(f"Profile: {profile}", "good"),
        pill("Screening, not underwriting", "warn"),
        pill(f"Refresh: {now_utc_str()}", "warn"),
        pill(f"Listings: {int(len(df_all))}", "good"),
    ],
)

st.caption(
    "This page does **not** display a recommendation on screen. "
    "The memo PDF contains the Barzel synthesis and rationale (for committee-style discussion)."
)

# Internal scoring (hidden from UI)
scores_df = compute_scores_df(df_all, DISTRICTS)
ranking = rank_for_profile(scores_df, profile=profile)

reco = pick_recommendation(ranking)

# Show only coverage / credibility status (no district named)
top_n = int(ranking.iloc[0]["listings"]) if ranking is not None and not ranking.empty and "listings" in ranking.columns else 0
conf = str(ranking.iloc[0]["confidence"]) if ranking is not None and not ranking.empty and "confidence" in ranking.columns else "—"
can_issue = bool(reco.get("district") is not None)

metric_grid(
    [
        ("Top sample size (n)", str(top_n), "Listings used for the top-ranked district"),
        ("Confidence", conf, "Based on sample size thresholds"),
        ("Memo gate", "Enabled" if can_issue else "Disabled", "Memo conclusion is gated when sample size is too low"),
        ("Scope", "Marina / Business Bay / JVC", "This demo scope is fixed"),
    ]
)

st.divider()
st.markdown("### Export investment memo (PDF)")
st.caption("The PDF may include a recommended district if and only if the gate is satisfied.")

# Pick rows for the PDF memo (defensive defaults)
top_row = ranking.iloc[0] if ranking is not None and not ranking.empty else pd.Series(dtype=object)
fallback_row = ranking.iloc[1] if ranking is not None and len(ranking) > 1 else None
aggressive_row = ranking.iloc[2] if ranking is not None and len(ranking) > 2 else None

recommended_district = str(reco.get("district") or "—")

pdf_bytes = build_recommendation_memo_pdf(
    profile=profile,
    recommended_district=recommended_district,
    top_row=top_row,
    fallback_row=fallback_row,
    aggressive_row=aggressive_row,
    df_all=df_all,
    districts=DISTRICTS,
)

st.download_button(
    "Download Investment Memo (PDF)",
    data=pdf_bytes,
    file_name="barzel_investment_memo.pdf",
    mime="application/pdf",
)
