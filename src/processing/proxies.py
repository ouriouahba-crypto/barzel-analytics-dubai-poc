import pandas as pd

def proxies(df: pd.DataFrame) -> dict:
    p = df["price_per_sqm"].dropna()
    d = df["days_active"].dropna()

    return {
        "value_proxy": float(p.median()) if len(p) else None,
        "liquidity_proxy": float(d.median()) if len(d) else None,
        "dispersion_proxy": float(p.quantile(0.75) - p.quantile(0.25)) if len(p) else None,
    }
