# pages/4_Recommendation.py
import streamlit as st
import pandas as pd

from src.app.pdf import build_recommendation_memo_pdf
from src.app.guardrails import require_non_empty, now_utc_str, district_counts, warn_low_coverage
from src.db.db import get_engine
from src.processing.engine import compute_scores_df, rank_for_profile, pick_recommendation

from src.app.ui import inject_base_ui, hero, metric_grid, pill

inject_base_ui()

DISTRICTS = ["Marina", "Business Bay", "JVC"]


@st.cache_data(ttl=60)
def load_listings() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql("select * from listings", engine)

    for col in ["price", "size_sqm", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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


df_all = load_listings()
require_non_empty(df_all, "listings")

warn_low_coverage(df_all, "price", 0.70, "price")
warn_low_coverage(df_all, "size_sqm", 0.60, "size_sqm")

profile = st.selectbox(
    "Investor profile",
    ["Capital Preservation", "Core", "Core+", "Opportunistic"],
    index=0,
)

scores_df = compute_scores_df(df_all, districts=DISTRICTS)
if scores_df is None or scores_df.empty:
    hero("Recommendation", "Profile-based district pick (screening, not underwriting).", pills=[pill("Scores error", "bad")])
    st.error("Scores could not be computed. Check numeric columns (price, size_sqm, timestamps).")
    st.stop()

ranked = rank_for_profile(scores_df, profile)
if ranked is None or ranked.empty:
    hero("Recommendation", "Profile-based district pick (screening, not underwriting).", pills=[pill("Ranking error", "bad")])
    st.error("Ranking returned empty. Verify scoring pipeline.")
    st.stop()

reco = pick_recommendation(scores_df, profile)

top = ranked.iloc[0]
runner = ranked.iloc[1] if len(ranked) > 1 else None

# Upside option: best value proxy among recommendable alternatives (if possible)
upside = None
try:
    tmp = ranked.copy()
    if "can_recommend" in tmp.columns:
        tmp = tmp[tmp["can_recommend"] == True]
    tmp = tmp[tmp["district"] != top["district"]]
    if not tmp.empty:
        upside = tmp.sort_values("yield", ascending=False).iloc[0]
except Exception:
    upside = None

hero(
    "Recommendation",
    "Capital allocation screening: one decision + shortlist + rationale.",
    pills=[
        pill(f"Profile: {profile}", "good"),
        pill("Screening, not underwriting", "warn"),
        pill("Risk Index: higher = more risky", "warn"),
        pill("Yield: value proxy", "warn"),
        pill(f"Refresh: {now_utc_str()}", "good"),
    ],
)

st.caption(
    "Interpretation: higher is better for Yield Potential and Adoption Speed. "
    "Risk Index is shown raw (higher = more risky) and inverted inside Profile Score."
)

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
    recommended_district = "—"
else:
    recommended_district = str(reco["district"])
    st.success(f"Decision (screening): Allocate focus to **{recommended_district}** for **{profile}**.")
    st.caption(f"Confidence: **{reco.get('confidence','—')}** (n={reco.get('listings','—')} listings)")

# -------------------------
# Capital guidance + next step (NOT advice, process guidance)
# -------------------------
st.divider()
st.subheader("Capital Guidance (screening)")
if recommended_district == "—":
    st.write("• Decision is gated due to insufficient coverage. Use this page as a framework, not a call.")
else:
    st.write(f"• **Primary focus district:** {recommended_district}")
    st.write("• **Use-case:** shortlist candidate district(s) before underwriting.")
    st.write("• **Next step if approved:** request deeper data and run underwriting-grade checks:")

    st.markdown(
        """
- Rent comps & true yield (net / gross)  
- Transaction data (not listings only)  
- Absorption / leasing velocity by sub-area  
- Supply pipeline & new deliveries  
- Unit mix, building quality, fees, service charges  
- Sensitivity: price, vacancy, capex, interest rate
        """
    )

# -------------------------
# Top card (top-ranked for transparency)
# -------------------------
st.subheader(f"Top-ranked (by Profile Score): **{top['district']}**")
metric_grid(
    [
        ("Profile Score", fmt_score(top["profile_score"]), "Higher is better (risk-inverted)"),
        ("Barzel Score", fmt_score(top["barzel"]), "0–100 composite"),
        ("Adoption Speed", fmt_score(top["adoption"]), "Liquidity proxy"),
        ("Risk Index", fmt_score(top["risk"]), "Higher = more risky"),
    ]
)

# -------------------------
# Shortlist
# -------------------------
st.divider()
st.markdown("### Shortlist (memo-ready)")

colA, colB, colC = st.columns(3)

with colA:
    st.markdown(f"**#1 — Top-ranked: {top['district']}**")
    st.metric("Profile Score", fmt_score(top["profile_score"]))
    st.caption(
        f"Barzel {fmt_score(top['barzel'])} · Value {fmt_score(top['yield'])} · "
        f"Liquidity {fmt_score(top['adoption'])} · Risk {fmt_score(top['risk'])}"
    )
    st.markdown("- Best overall profile fit\n- Primary screening candidate\n- Decision is gated by coverage")

with colB:
    if runner is None:
        st.info("No fallback option available.")
    else:
        st.markdown(f"**#2 — Fallback: {runner['district']}**")
        st.metric("Profile Score", fmt_score(runner["profile_score"]))
        st.caption(
            f"Barzel {fmt_score(runner['barzel'])} · Value {fmt_score(runner['yield'])} · "
            f"Liquidity {fmt_score(runner['adoption'])} · Risk {fmt_score(runner['risk'])}"
        )
        st.markdown("- Second-best profile fit\n- Use if #1 fails constraints\n- Trade-off alternative")

with colC:
    if upside is None:
        st.info("No upside option available.")
    else:
        st.markdown(f"**#3 — Upside option: {upside['district']}**")
        st.metric("Value Proxy", fmt_score(upside["yield"]))
        st.caption(
            f"Profile {fmt_score(upside['profile_score'])} · Barzel {fmt_score(upside['barzel'])} · "
            f"Liquidity {fmt_score(upside['adoption'])} · Risk {fmt_score(upside['risk'])}"
        )
        st.markdown("- Value-max bias (price/sqm)\n- Accepts risk/dispersion\n- Opportunistic sleeve only")

# -------------------------
# Rationale vs #2 (3 bullets)
# -------------------------
st.divider()
st.markdown("### Decision rationale (vs #2)")

if runner is None:
    st.write("Not enough alternatives to compute a meaningful rationale.")
else:
    if profile == "Capital Preservation":
        driver_names = ["risk", "adoption", "depth"]
    elif profile == "Core":
        driver_names = ["barzel", "adoption", "risk"]
    elif profile == "Core+":
        driver_names = ["yield", "adoption", "risk"]
    else:  # Opportunistic
        driver_names = ["yield", "adoption", "momentum"]

    labels = {
        "barzel": "Barzel Score",
        "yield": "Value Proxy (price/sqm)",
        "adoption": "Liquidity Proxy (Adoption Speed)",
        "risk": "Risk Index (higher = more risky)",
        "momentum": "Momentum (recency proxy)",
        "depth": "Market Depth (coverage proxy)",
    }

    bullets = []
    for k in driver_names:
        if k not in top or k not in runner:
            continue
        bullets.append((k, labels[k], float(top[k]), float(runner[k])))
        if len(bullets) >= 2:
            break

    candidates = ["yield", "adoption", "risk", "momentum", "depth", "barzel"]
    trade_k = None
    trade_val = None
    for k in candidates:
        if k not in top or k not in runner:
            continue
        if any(b[0] == k for b in bullets):
            continue
        dv = abs(float(top[k]) - float(runner[k]))
        if trade_val is None or dv > trade_val:
            trade_val = dv
            trade_k = k

    st.markdown("**Top drivers**")
    for k, name, a, b in bullets:
        extra = " (lower is better)" if k == "risk" else ""
        st.write(f"• **{name}**: {top['district']} = **{fmt_score(a)}** vs {runner['district']} = **{fmt_score(b)}** (Δ {delta_str(a, b)}){extra}")

    if trade_k is not None:
        a = float(top[trade_k])
        b = float(runner[trade_k])
        st.markdown("**Main trade-off**")
        st.write(f"• **{labels[trade_k]}**: {top['district']} = **{fmt_score(a)}** vs {runner['district']} = **{fmt_score(b)}** (Δ {delta_str(a, b)})")

# -------------------------
# Table + provenance + export
# -------------------------
st.divider()
st.markdown("### Alternatives (ranked)")

show_cols = ["district", "profile_score", "barzel", "yield", "adoption", "risk", "depth", "momentum", "listings"]
for extra in ["confidence", "can_recommend"]:
    if extra in ranked.columns:
        show_cols.append(extra)

show_cols = [c for c in show_cols if c in ranked.columns]
st.dataframe(ranked[show_cols], use_container_width=True)
st.caption("Risk is inverted inside Profile Score. Risk Index is displayed raw (higher = more risky).")

st.divider()
st.markdown("### Data provenance")
counts = district_counts(df_all, DISTRICTS)
st.caption(
    f"Data refreshed: **{now_utc_str()}** · Total listings: **{len(df_all)}** · "
    f"Marina: **{counts['Marina']}** | Business Bay: **{counts['Business Bay']}** | JVC: **{counts['JVC']}**"
)

st.divider()
st.markdown("### Export")

pdf_bytes = build_recommendation_memo_pdf(
    profile=profile,
    recommended_district=recommended_district,
    top_row=top,
    fallback_row=runner,
    aggressive_row=upside,
    df_all=df_all,
)

safe_profile = profile.lower().replace(" ", "_").replace("-", "_")
safe_reco = (recommended_district if recommended_district != "—" else "no_reco").lower().replace(" ", "_").replace("-", "_")

st.download_button(
    label="📄 Download 1-page Recommendation Memo (PDF)",
    data=pdf_bytes,
    file_name=f"barzel_reco_memo_{safe_reco}_{safe_profile}.pdf",
    mime="application/pdf",
)
