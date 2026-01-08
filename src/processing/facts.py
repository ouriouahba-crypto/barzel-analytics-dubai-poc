import pandas as pd

def market_facts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["price"] = pd.to_numeric(out.get("price"), errors="coerce")
    out["size_sqm"] = pd.to_numeric(out.get("size_sqm"), errors="coerce")

    out["price_per_sqm"] = out["price"] / out["size_sqm"]
    out.loc[(out["price_per_sqm"] < 1000) | (out["price_per_sqm"] > 25000), "price_per_sqm"] = None

    for c in ["first_seen", "last_seen"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", utc=True)

    now = pd.Timestamp.utcnow()
    if "first_seen" in out.columns:
        out["days_active"] = (out["last_seen"].fillna(now) - out["first_seen"]).dt.days
        out.loc[(out["days_active"] < 0) | (out["days_active"] > 3650), "days_active"] = None

    return out
