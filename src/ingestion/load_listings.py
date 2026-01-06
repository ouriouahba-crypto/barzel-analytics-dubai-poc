# src/ingestion/load_listings.py
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from sqlalchemy import text

from src.db.db import get_engine


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def table_exists(conn, table_name: str) -> bool:
    # works on Postgres (Neon)
    return conn.execute(text("SELECT to_regclass(:t)"), {"t": f"public.{table_name}"}).scalar() is not None


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/listings_enriched.csv",
        help="Path to CSV to load (default: data/listings_enriched.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=["replace", "truncate", "append"],
        default="replace",
        help="replace = drop+recreate table, truncate = empty then insert, append = insert only",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    engine = get_engine()

    logging.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    if "id" not in df.columns:
        raise SystemExit("CSV must contain an 'id' column (primary key).")

    # Basic hygiene: drop dup ids inside the file
    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first").copy()
    logging.info("Dedup by id: %d -> %d", before, len(df))

    # Parse dates if present
    for c in ["first_seen", "last_seen", "start_date", "end_date", "scraped_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # If user asked truncate but table doesn't exist, fallback to replace
    mode = args.mode
    with engine.begin() as conn:
        exists = table_exists(conn, "listings")
        if mode == "truncate" and not exists:
            logging.warning("Table listings does not exist -> switching mode truncate -> replace")
            mode = "replace"

        if mode == "truncate":
            logging.info("TRUNCATE listings on Neon...")
            conn.execute(text("TRUNCATE TABLE listings"))

    # Write to DB
    if_exists = "replace" if mode == "replace" else "append"
    logging.info("Writing to Neon: table=listings if_exists=%s rows=%d", if_exists, len(df))

    # method="multi" speeds up inserts a lot
    df.to_sql(
        "listings",
        engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )

    # Re-add PK after replace (optional but clean)
    if mode == "replace":
        with engine.begin() as conn:
            try:
                conn.execute(text("ALTER TABLE listings ADD PRIMARY KEY (id)"))
                logging.info("Primary key restored: listings(id)")
            except Exception as e:
                logging.warning("Could not add primary key (maybe already exists). Error: %s", e)

    # Final count
    with engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM listings")).scalar()
        logging.info("Done. rows in neon.listings = %s", n)


if __name__ == "__main__":
    main()
