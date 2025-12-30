# pages/2_Compare.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text

from src.app.pdf import build_compare_memo_pdf
from src.app.guardrails import require_non_empty, now_utc_str, district_counts, warn_low_coverage
from src.db.db import get_engine
from src.processing.engine import compute_scores_df


from src.app.ui import inject_base_ui, hero, metric_grid, pill
inject_base_ui()

DISTRICTS = ["Marina", "Business Bay", "JVC"]


@st.cache_data(ttl=120)
def load_data() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM listings"), conn)
    return df


def tradeoff_signal(row: pd.Series) -> float:
    """
    Trade-off score where higher is better.
    We reward Yield Potential + Adoption Speed,
    and we reward LOW risk by inverting it: (100 - risk).
    """
    y = float(row.get("Yield Potential", 0.0))
    a = float(row.get("Adoption Speed", 0.0))
    r = float(row.get("Risk Index", 0.0))  # higher = more risky
    # weights are intentionally simple for demo
    return (0.45 * y) + (0.45 * a) + (0.10 * (100.0 - r))


df = load_data()
require_non_empty(df, "listings")

warn_low_coverage(df, "price", 0.70, "price")
warn_low_coverage(df, "size_sqm", 0.60, "size_sqm")

hero(
    "Compare",
    "Yield vs Liquidity trade-off view (risk-aware screening).",
    pills=[
        pill("Risk Index: higher = more risky", "warn"),
        pill("Yield: value proxy", "warn"),
        pill(f"Refresh: {now_utc_str()}", "good"),
    ],
)

scores_df = compute_scores_df(df, districts=DISTRICTS)
if scores_df is None or scores_df.empty:
    st.error("Scores could not be computed. Check numeric columns (price, size_sqm, timestamps).")
    st.stop()

compare_df = scores_df.rename(
    columns={
        "district": "District",
        "yield": "Yield Potential",
        "adoption": "Adoption Speed",
        "risk": "Risk Index",
        "barzel": "Barzel Score",
    }
).copy()

# Keep only what we need + carry confidence signals
keep_cols = ["District", "Yield Potential", "Adoption Speed", "Risk Index", "Barzel Score", "listings"]
for extra in ["confidence", "can_recommend", "momentum", "depth"]:
    if extra in compare_df.columns:
        keep_cols.append(extra)

compare_df = compare_df[keep_cols].copy()

for ccol in ["Yield Potential", "Adoption Speed", "Risk Index", "Barzel Score"]:
    compare_df[ccol] = pd.to_numeric(compare_df[ccol], errors="coerce").fillna(0.0)

# Trade-off signal (risk-aware)
compare_df["Trade-off Score"] = compare_df.apply(tradeoff_signal, axis=1).round(1)

# Rank: recommendable first, then by score
if "can_recommend" in compare_df.columns:
    compare_df["_penalty"] = compare_df["can_recommend"].apply(lambda x: 0 if bool(x) else 1000)
else:
    compare_df["_penalty"] = 0

ranked = compare_df.sort_values(["_penalty", "Trade-off Score"], ascending=[True, False]).copy()
top = ranked.iloc[0]

# Decision banner (gated)
recommended = str(top["District"])
can_reco = bool(top.get("can_recommend", True))
conf = str(top.get("confidence", "—"))
n = int(top.get("listings", 0))

if not can_reco:
    st.warning(
        f"No recommendation: insufficient coverage (min listings required). "
        f"Top candidate by trade-off score is **{recommended}** (n={n}, confidence={conf})."
    )
    st.caption("We still show the trade-off view for transparency, but we do not output a decision under low coverage.")
else:
    st.success(
        f"Decision (trade-off screening): **{recommended}** offers the strongest Yield/Liquidity mix "
        f"with risk-awareness (n={n}, confidence={conf})."
    )

metric_grid(
    [
        ("Top candidate", recommended, "Trade-off score ranking"),
        ("Yield Potential", f"{float(top['Yield Potential']):.1f}", "Value proxy (price/sqm)"),
        ("Adoption Speed", f"{float(top['Adoption Speed']):.1f}", "Liquidity proxy"),
        ("Risk Index", f"{float(top['Risk Index']):.1f}", "Higher = more dispersion (more risk)"),
    ]
)

st.caption(
    "Interpretation: higher is better for Yield Potential and Adoption Speed. "
    "Risk Index is inverted inside the trade-off score (safer districts score higher)."
)

fig = px.scatter(
    compare_df,
    x="Adoption Speed",
    y="Yield Potential",
    size="Barzel Score",
    color="District",
    hover_name="District",
    size_max=60,
    title="Yield Potential (Value proxy) vs Adoption Speed (Liquidity proxy)",
)

fig.update_layout(
    xaxis_title="Adoption Speed (Liquidity proxy)",
    yaxis_title="Yield Potential (Value proxy)",
    legend_title_text="District",
    xaxis=dict(range=[0, 100]),
    yaxis=dict(range=[0, 100]),
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Each point represents a district. Upper-right suggests stronger yield potential with faster absorption (proxy).")

st.subheader("KPI Comparison")
display_df = compare_df.drop(columns=["_penalty"]).set_index("District")
st.dataframe(display_df, use_container_width=True)

st.divider()
st.markdown("### Data provenance")
counts = district_counts(df, DISTRICTS)
st.caption(
    f"Data refreshed: **{now_utc_str()}** · Total listings: **{len(df)}** · "
    f"Marina: **{counts['Marina']}** | Business Bay: **{counts['Business Bay']}** | JVC: **{counts['JVC']}**"
)

st.divider()
st.markdown("### Export")

pdf_bytes = build_compare_memo_pdf(
    compare_df=compare_df.drop(columns=["_penalty"]),
    recommended=recommended if can_reco else "—",
    df_all=df,
)

safe_reco = (recommended if can_reco else "no_reco").lower().replace(" ", "_").replace("-", "_")
st.download_button(
    label="📄 Download 1-page Compare Memo (PDF)",
    data=pdf_bytes,
    file_name=f"barzel_compare_memo_{safe_reco}.pdf",
    mime="application/pdf",
)
