def quality(df):
    n = len(df)
    coverage = {
        "price_per_sqm": df["price_per_sqm"].notna().mean(),
        "days_active": df["days_active"].notna().mean(),
    }
    return {"n": n, "coverage": coverage}
