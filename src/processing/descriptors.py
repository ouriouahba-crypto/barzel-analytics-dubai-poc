import pandas as pd

def descriptors(df: pd.DataFrame) -> dict:
    p = df["price_per_sqm"].dropna()
    d = df["days_active"].dropna()

    return {
        "price_p25": p.quantile(0.25) if len(p) else None,
        "price_p50": p.quantile(0.50) if len(p) else None,
        "price_p75": p.quantile(0.75) if len(p) else None,
        "price_iqr": (p.quantile(0.75) - p.quantile(0.25)) if len(p) else None,
        "dom_p50": d.quantile(0.50) if len(d) else None,
        "dom_p75": d.quantile(0.75) if len(d) else None,
        "listings": int(len(df)),
    }
