import pandas as pd
from src.db.db import get_engine

CSV_PATH = "data/raw/dubai_listings.csv"

EXPECTED_COLS = [
    "id","source","district","price","size_sqm","bedrooms","property_type",
    "latitude","longitude","first_seen","last_seen"
]

def main():
    df = pd.read_csv(CSV_PATH)

    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df = df[EXPECTED_COLS].copy()

    # basic cleanup
    df["district"] = df["district"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df["property_type"] = df["property_type"].astype(str).str.strip()

    # force numeric types (important for KPIs)
    for col in ["price", "size_sqm", "bedrooms", "latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # force datetime
    df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce")
    df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce")

    # fail fast if critical columns are broken
    if df["price"].isna().mean() > 0.10 or df["size_sqm"].isna().mean() > 0.10:
        raise ValueError("Too many NaNs in price/size_sqm after parsing. Check CSV formatting.")

    engine = get_engine()

    df.to_sql("listings", engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into listings")

if __name__ == "__main__":
    main()
