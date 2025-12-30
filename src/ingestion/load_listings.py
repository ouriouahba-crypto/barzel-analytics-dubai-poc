import pandas as pd
from src.db.db import get_engine

CSV_PATH = "data/raw/dubai_listings.csv"

EXPECTED_COLS = [
    "id","source","district","price","size_sqm","bedrooms","property_type",
    "latitude","longitude","first_seen","last_seen"
]

def main():
    df = pd.read_csv(CSV_PATH)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate columns
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Keep only expected columns in correct order
    df = df[EXPECTED_COLS]

    # Basic cleanup
    df["district"] = df["district"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df["property_type"] = df["property_type"].astype(str).str.strip()

    engine = get_engine()

    # Insert (append). If duplicate id exists, it will fail (primary key).
    df.to_sql("listings", engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into listings")

if __name__ == "__main__":
    main()
