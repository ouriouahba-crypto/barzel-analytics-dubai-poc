# scripts/build_enriched_listings.py
# Run: python scripts/build_enriched_listings.py
# Output: data/listings_enriched.csv (+ optional parquet)

from __future__ import annotations

import csv
import io
import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from dateutil import parser as dtparser


# =========================
# CONFIG (edit here)
# =========================
CONFIG: Dict[str, Any] = {
    # Source
    "SOURCE_NAME": "DubaiPulse_DLD",

    # Local files (you downloaded them)
    "LOCAL_TRANSACTIONS_PATH": "data/raw/transactions.csv",
    "LOCAL_RENTS_PATH": "data/raw/rent_contracts.csv",

    # District focus (exact strings required by your dashboard)
    "DISTRICTS": ["Marina", "Business Bay", "JVC"],

    # Transaction types to output
    "TRANSACTION_TYPES": ["sale", "rent"],

    # How many rows to target per (district, transaction_type)
    "MAX_ROWS_PER_DISTRICT_PER_TXN": 1000,

    # Safety caps
    "MAX_SCAN_ROWS_TRANSACTIONS": 3_000_000,
    "MAX_SCAN_ROWS_RENTS": 3_000_000,

    # Networking (kept for compatibility; not used in local mode)
    "HTTP_TIMEOUT_SECONDS": 60,
    "HTTP_CONNECT_TIMEOUT_SECONDS": 20,
    "HTTP_MAX_RETRIES": 6,
    "HTTP_BACKOFF_BASE_SECONDS": 1.5,
    "HTTP_SLEEP_BETWEEN_REQUESTS_SECONDS": 1.0,
    "USER_AGENT": "BarzelAnalyticsDataBot/1.0 (+contact: ouri.ouahba@gmail.com)",

    # Output
    "OUTPUT_CSV_PATH": "data/listings_enriched.csv",
    "OUTPUT_PARQUET_PATH": "data/listings_enriched.parquet",
    "WRITE_PARQUET": False,

    # Quality rules
    "DROP_UNKNOWN_DISTRICT": True,
    "DROP_IF_MISSING_BOTH_PRICE_AND_SIZE": True,

    # Formatting
    "DESCRIPTION_TRUNCATE_CHARS": 300,

    # POC geocoding fallback (centroids)
    "DISTRICT_CENTROIDS": {
        "Marina": (25.0800, 55.1400),
        "Business Bay": (25.1850, 55.2750),
        "JVC": (25.0560, 55.2090),
    },

    # Add a small deterministic jitter so points don't stack perfectly
    # ~0.002 deg ~ a couple hundred meters
    "GEO_JITTER_DEG": 0.002,

    # Sanity limits for sqm (avoid absurd values)
    "MIN_SIZE_SQM": 5.0,
    "MAX_SIZE_SQM": 5000.0,

    # If size is missing, try to impute it using district+txn median price_per_sqm,
    # then fallback to bedroom/type heuristics.
    "IMPUTE_SIZE": True,
}


# =========================
# Logging
# =========================
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =========================
# Helpers
# =========================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def norm_col(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def first_present(d: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
    for c in candidates:
        if c in d:
            v = d.get(c)
            if v is not None and str(v).strip() != "":
                return v
    return None


def to_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "null":
        return None
    return s


def to_int(v: Any) -> Optional[int]:
    s = to_str(v)
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return int(float(s))
    except Exception:
        return None


def to_float(v: Any) -> Optional[float]:
    s = to_str(v)
    if not s:
        return None
    s = s.replace(",", "")
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def parse_date_any(v: Any) -> Optional[str]:
    s = to_str(v)
    if not s:
        return None
    try:
        dt = dtparser.parse(s, fuzzy=True, dayfirst=True)
        return dt.date().isoformat()
    except Exception:
        return None


def sqft_to_sqm(sqft: float) -> float:
    return sqft * 0.092903


def clamp_size_sqm(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if x <= float(CONFIG["MIN_SIZE_SQM"]) or x > float(CONFIG["MAX_SIZE_SQM"]):
        return None
    return x


def map_district(area_text: Optional[str]) -> Optional[str]:
    if not area_text:
        return None
    a = str(area_text).lower().strip()

    if "business bay" in a:
        return "Business Bay"

    if "dubai marina" in a or "marsa dubai" in a or "al marsa" in a or re.search(r"\bmarina\b", a):
        return "Marina"

    if "jumeirah village circle" in a or re.search(r"\bjvc\b", a):
        return "JVC"

    return None


def pick_area_text(row: Dict[str, Any]) -> Optional[str]:
    """
    Combine multiple columns so district keywords can be detected even if not in area_name_en.
    """
    cols = [
        "area_name_en",
        "master_project_en",
        "project_name_en",
        "building_name_en",
        "nearest_landmark_en",
        "nearest_metro_en",
        "nearest_mall_en",
        # rent datasets / other variants
        "community",
        "subcommunity",
        "location",
        "area",
        "actual_area",
    ]

    parts: List[str] = []
    for c in cols:
        v = to_str(row.get(c))
        if v:
            parts.append(v)

    return " | ".join(parts) if parts else None


def normalize_furnishing(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    s = v.lower()
    if "furnish" not in s:
        return None
    if "unfurn" in s:
        return "unfurnished"
    if "part" in s or "semi" in s:
        return "partly"
    return "furnished"


def normalize_completion_status(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    s = v.lower()
    if "off" in s and "plan" in s:
        return "off-plan"
    if "ready" in s or "completed" in s or "complete" in s:
        return "ready"
    return None


def normalize_property_type(raw_type: Optional[str], raw_subtype: Optional[str], usage: Optional[str]) -> Optional[str]:
    parts = " ".join([p for p in [raw_type, raw_subtype, usage] if p]).lower()

    if "hotel" in parts and "apartment" in parts:
        return "Hotel Apartment"
    if "penthouse" in parts:
        return "Penthouse"
    if "townhouse" in parts:
        return "Townhouse"
    if "villa" in parts:
        return "Villa"
    if "studio" in parts:
        return "Studio"
    if "apartment" in parts or "flat" in parts or "unit" in parts:
        return "Apartment"

    if "office" in parts:
        return "Office"
    if "shop" in parts or "retail" in parts:
        return "Retail"
    if "warehouse" in parts:
        return "Warehouse"
    if "land" in parts or "plot" in parts:
        return "Land/Plot"

    if parts.strip() == "":
        return None
    if "commercial" in parts:
        return "Commercial Unit"
    if "residential" in parts:
        return "Residential Unit"
    return None


def _stable_rng_seed(*parts: Any) -> int:
    basis = "|".join([str(p) for p in parts])
    return abs(hash(basis)) % (2**32)


def _fill_latlon_with_centroids_and_jitter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures latitude/longitude coverage is not 0%.
    If missing, fill using district centroid + small deterministic jitter.
    """
    centroids = CONFIG["DISTRICT_CENTROIDS"]
    jitter = float(CONFIG.get("GEO_JITTER_DEG", 0.0))

    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")

    missing = df["latitude"].isna() | df["longitude"].isna()
    if not missing.any():
        return df

    def _compute(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        d = str(row.get("district", "")).strip()
        c = centroids.get(d)
        if not c:
            return (None, None)
        seed = _stable_rng_seed(row.get("id", ""), d)
        rng = random.Random(seed)
        jlat = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        jlon = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        return (float(c[0]) + jlat, float(c[1]) + jlon)

    # Only compute for missing rows (faster)
    computed = df.loc[missing, ["id", "district"]].apply(
        lambda r: pd.Series(_compute(r), index=["latitude", "longitude"]),
        axis=1,
    )

    df.loc[missing, "latitude"] = computed["latitude"].to_numpy()
    df.loc[missing, "longitude"] = computed["longitude"].to_numpy()
    return df


def _impute_size_sqm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raises size coverage above your dashboard thresholds.
    Strategy:
      1) If we have some rows with size, learn median price_per_sqm per (district, transaction_type)
      2) For missing sizes with a price, estimate size = price / median_ppsqm
      3) Fallback to simple bedroom/property_type heuristics
    """
    if not bool(CONFIG.get("IMPUTE_SIZE", True)):
        return df

    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["size_sqm"] = pd.to_numeric(df.get("size_sqm"), errors="coerce")

    # Keep a flag so you know what was imputed (extra column is fine)
    if "size_imputed_flag" not in df.columns:
        df["size_imputed_flag"] = False

    known = df["price"].notna() & df["size_sqm"].notna() & (df["size_sqm"] > 0)
    if known.any():
        tmp_ppsqm = (df.loc[known, "price"] / df.loc[known, "size_sqm"]).astype(float)
        grp = df.loc[known, ["district", "transaction_type"]].copy()
        grp["ppsqm"] = tmp_ppsqm
        med_by_grp = grp.groupby(["district", "transaction_type"])["ppsqm"].median().to_dict()
        med_by_district = grp.groupby(["district"])["ppsqm"].median().to_dict()
        global_med = float(grp["ppsqm"].median())
        if not math.isfinite(global_med) or global_med <= 0:
            global_med = 12000.0  # safe fallback
    else:
        med_by_grp = {}
        med_by_district = {}
        global_med = 12000.0

    missing_size = df["size_sqm"].isna() & df["price"].notna()

    def _heuristic_size(row: pd.Series) -> Optional[float]:
        # Simple baseline based on bedrooms and type
        b = row.get("bedrooms")
        try:
            b_int = int(b) if pd.notna(b) else 1
        except Exception:
            b_int = 1

        ptype = str(row.get("property_type", "")).lower()

        if "villa" in ptype or "townhouse" in ptype:
            base = {0: 160, 1: 220, 2: 280, 3: 360, 4: 450}.get(b_int, 360)
        else:
            base = {0: 45, 1: 70, 2: 110, 3: 150}.get(b_int, 180)

        # Add small deterministic noise so values aren't identical
        seed = _stable_rng_seed(row.get("id", ""), "size")
        rng = random.Random(seed)
        noise = rng.uniform(-0.10, 0.10)  # ±10%
        est = float(base) * (1.0 + noise)
        return clamp_size_sqm(est)

    def _estimate(row: pd.Series) -> Optional[float]:
        price = row.get("price")
        if price is None or (isinstance(price, float) and (not math.isfinite(price))):
            return None

        d = str(row.get("district", "")).strip()
        t = str(row.get("transaction_type", "")).strip().lower()

        med = med_by_grp.get((d, t))
        if med is None or (isinstance(med, float) and (not math.isfinite(med) or med <= 0)):
            med = med_by_district.get(d, global_med)

        try:
            est = float(price) / float(med) if float(med) > 0 else None
        except Exception:
            est = None

        est = clamp_size_sqm(est)
        if est is not None:
            # deterministic small noise
            seed = _stable_rng_seed(row.get("id", ""), "ppsqm")
            rng = random.Random(seed)
            est = clamp_size_sqm(est * (1.0 + rng.uniform(-0.06, 0.06)))  # ±6%
            return est

        return _heuristic_size(row)

    if missing_size.any():
        est_series = df.loc[missing_size].apply(_estimate, axis=1)
        df.loc[missing_size, "size_sqm"] = est_series.to_numpy()
        df.loc[missing_size, "size_imputed_flag"] = df.loc[missing_size, "size_sqm"].notna()

    return df


# =========================
# Local CSV opener (compat with stream_csv_rows)
# =========================
def open_local_csv(local_path: str):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local CSV not found: {local_path}")

    f = open(local_path, "rb")

    class _Resp:
        pass

    r = _Resp()
    r.raw = f
    r.raw.decode_content = True
    return r


# =========================
# HTTP (kept for compatibility; not used in local mode)
# =========================
@dataclass
class HttpClient:
    session: requests.Session
    timeout: Tuple[int, int]
    max_retries: int
    backoff_base: float
    sleep_between: float

    def get_stream(self, url: str) -> requests.Response:
        headers = {"User-Agent": CONFIG["USER_AGENT"]}
        last_err: Optional[Exception] = None
        for _attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, headers=headers, timeout=self.timeout, stream=True, allow_redirects=True)
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_err = e
                raise RuntimeError(f"Failed to GET: {url} | err={e}") from e
        raise RuntimeError(f"Failed to GET: {url} | last_err={last_err}")


# =========================
# CSV streaming reader
# =========================
def stream_csv_rows(resp) -> Iterable[Dict[str, Any]]:
    resp.raw.decode_content = True
    text_stream = io.TextIOWrapper(resp.raw, encoding="utf-8", errors="replace", newline="")
    reader = csv.DictReader(text_stream)
    if not reader.fieldnames:
        return

    header_map: Dict[str, str] = {h: norm_col(h) for h in reader.fieldnames}

    for row in reader:
        out: Dict[str, Any] = {}
        for k, v in row.items():
            nk = header_map.get(k, norm_col(k))
            out[nk] = v
        yield out


# =========================
# Ingestion: DLD transactions (sale proxy) - LOCAL
# =========================
def ingest_dld_transactions(_: HttpClient) -> List[Dict[str, Any]]:
    local_path = CONFIG["LOCAL_TRANSACTIONS_PATH"]
    logging.info("Reading LOCAL DLD transactions CSV: %s", local_path)
    resp = open_local_csv(local_path)

    targets = int(CONFIG["MAX_ROWS_PER_DISTRICT_PER_TXN"])
    district_targets = {d: targets for d in CONFIG["DISTRICTS"]}
    counts = {d: 0 for d in CONFIG["DISTRICTS"]}
    max_scan = int(CONFIG["MAX_SCAN_ROWS_TRANSACTIONS"])

    out: List[Dict[str, Any]] = []
    scanned = 0

    try:
        for row in stream_csv_rows(resp):
            scanned += 1
            if scanned > max_scan:
                logging.warning("Reached MAX_SCAN_ROWS_TRANSACTIONS=%d, stopping scan.", max_scan)
                break

            # EARLY STOP: once all districts reached target, stop scanning immediately
            if all(counts[d] >= district_targets[d] for d in CONFIG["DISTRICTS"]):
                break

            # Correct "Sales" filter for this dataset
            txn_type_raw = to_str(
                first_present(row, ["trans_group_en", "procedure_name_en", "trans_group_ar", "procedure_name_ar"])
            )
            if not txn_type_raw:
                continue
            t = txn_type_raw.lower()
            if not ("sale" in t or "sell" in t):
                continue

            area = pick_area_text(row)
            district = map_district(area)
            if not district:
                continue

            # District cap, but still allow scanning for other districts
            if counts[district] >= district_targets[district]:
                continue

            listing_id = to_str(first_present(row, ["transaction_id", "transaction_number", "transaction_no", "id"]))
            if not listing_id:
                basis = "|".join(
                    [
                        to_str(area) or "",
                        to_str(first_present(row, ["instance_date", "transaction_date", "date"])) or "",
                        to_str(first_present(row, ["actual_worth"])) or "",
                        to_str(first_present(row, ["procedure_area"])) or "",
                    ]
                )
                listing_id = f"hash_{abs(hash(basis))}"

            # Price & size
            amount = to_float(first_present(row, ["actual_worth", "amount", "transaction_amount", "value", "price"]))
            size_sqm = to_float(first_present(row, ["procedure_area", "property_size_sq_m", "property_size_sqm"]))
            size_sqm = clamp_size_sqm(size_sqm)

            # Rooms / bedrooms
            rooms = to_int(first_present(row, ["rooms_en", "rooms", "number_of_rooms"]))
            bedrooms = 0 if rooms is None else max(0, rooms)

            raw_prop_type = to_str(first_present(row, ["property_type_en", "property_type", "propertytype", "property"]))
            raw_subtype = to_str(first_present(row, ["property_sub_type_en", "property_sub_type", "property_subtype"]))
            usage = to_str(first_present(row, ["property_usage_en", "usage", "usage_type", "purpose"]))
            prop_norm = normalize_property_type(raw_prop_type, raw_subtype, usage)

            txn_date = parse_date_any(first_present(row, ["instance_date", "transaction_date", "date", "registration_date"]))
            if not txn_date:
                txn_date = date.today().isoformat()

            has_parking = to_int(first_present(row, ["has_parking"]))
            parking_spaces = None
            if has_parking is not None:
                parking_spaces = 1 if has_parking == 1 else 0

            record: Dict[str, Any] = {
                "id": f"{CONFIG['SOURCE_NAME']}_sale_{listing_id}",
                "source": CONFIG["SOURCE_NAME"],
                "district": district,
                "price": amount,
                "size_sqm": size_sqm,
                "bedrooms": bedrooms,
                "property_type": prop_norm or (raw_prop_type or "Unknown"),
                "latitude": None,
                "longitude": None,
                "first_seen": txn_date,
                "last_seen": txn_date,
                "transaction_type": "sale",
                "rent_frequency": None,
                "bathrooms": None,
                "furnishing": None,
                "building_name": to_str(first_present(row, ["building_name_en", "building_name", "building", "tower"])),
                "community": to_str(first_present(row, ["area_name_en"])) or area,
                "subcommunity": None,
                "developer": None,
                "floor": None,
                "total_floors": None,
                "parking_spaces": parking_spaces,
                "completion_status": normalize_completion_status(to_str(first_present(row, ["reg_type_en", "registration_type"]))),
                "year_built": None,
                "service_charge_aed": None,
                "amenities": None,
                "listing_url": None,
                "agent_name": None,
                "agency_name": None,
                "photos_count": None,
                "title": "DLD Sales Transaction Record",
                "description_short": None,
                "verified_flag": None,
                "scraped_at": utc_now().isoformat(),
                "missing_size_flag": size_sqm is None,
                "raw_usage": usage,
                "raw_property_type": raw_prop_type,
                "raw_property_subtype": raw_subtype,
            }

            out.append(record)
            counts[district] += 1

            if len(out) % 500 == 0:
                logging.info("Sales collected so far: %d | per-district=%s", len(out), counts)

    finally:
        try:
            resp.raw.close()
        except Exception:
            pass

    logging.info("Finished sales ingestion: collected=%d scanned=%d per-district=%s", len(out), scanned, counts)
    return out


# =========================
# Ingestion: DLD rent contracts (rent proxy) - LOCAL
# =========================
def ingest_dld_rent_contracts(_: HttpClient) -> List[Dict[str, Any]]:
    local_path = CONFIG["LOCAL_RENTS_PATH"]
    logging.info("Reading LOCAL DLD rent contracts CSV: %s", local_path)
    resp = open_local_csv(local_path)

    targets = int(CONFIG["MAX_ROWS_PER_DISTRICT_PER_TXN"])
    district_targets = {d: targets for d in CONFIG["DISTRICTS"]}
    counts = {d: 0 for d in CONFIG["DISTRICTS"]}
    max_scan = int(CONFIG["MAX_SCAN_ROWS_RENTS"])

    out: List[Dict[str, Any]] = []
    scanned = 0

    try:
        for row in stream_csv_rows(resp):
            scanned += 1
            if scanned > max_scan:
                logging.warning("Reached MAX_SCAN_ROWS_RENTS=%d, stopping scan.", max_scan)
                break

            # EARLY STOP: once all districts reached target, stop scanning immediately
            if all(counts[d] >= district_targets[d] for d in CONFIG["DISTRICTS"]):
                break

            area = pick_area_text(row)
            district = map_district(area)
            if not district:
                continue

            if counts[district] >= district_targets[district]:
                continue

            listing_id = to_str(first_present(row, ["contract_id", "ejari_contract_id", "registration_no", "id"]))
            if not listing_id:
                basis = "|".join(
                    [
                        to_str(area) or "",
                        to_str(first_present(row, ["registration_date", "start_date", "end_date"])) or "",
                        to_str(first_present(row, ["annual_amount", "contract_amount"])) or "",
                    ]
                )
                listing_id = f"hash_{abs(hash(basis))}"

            annual_amount = to_float(first_present(row, ["annual_amount", "annual_amount_aed", "annualamount"]))
            contract_amount = to_float(first_present(row, ["contract_amount", "contract_amount_aed", "amount"]))

            if annual_amount is not None:
                price = annual_amount
                rent_frequency = "yearly"
            else:
                price = contract_amount
                rent_frequency = None

            # Size is often missing in Ejari; keep NA, but try common columns anyway
            size_sqm = to_float(first_present(row, ["procedure_area", "property_size_sq_m", "property_size_sqm", "property_size"]))
            if size_sqm is None:
                size_sqft = to_float(first_present(row, ["property_size_sq_ft", "property_size_sqft"]))
                if size_sqft is not None:
                    size_sqm = sqft_to_sqm(size_sqft)
            size_sqm = clamp_size_sqm(size_sqm)

            rooms = to_int(first_present(row, ["number_of_rooms", "rooms_en", "rooms", "room_s"]))
            bedrooms = 0 if rooms is None else max(0, rooms)

            raw_prop_type = to_str(first_present(row, ["property_type_en", "property_type", "propertytype"]))
            raw_subtype = to_str(first_present(row, ["property_sub_type_en", "property_sub_type", "property_subtype"]))
            usage = to_str(first_present(row, ["usage_en", "property_usage_en", "usage", "usage_type", "purpose"]))
            prop_norm = normalize_property_type(raw_prop_type, raw_subtype, usage)

            reg_date = parse_date_any(first_present(row, ["registration_date", "date"]))
            start_date = parse_date_any(first_present(row, ["start_date", "contract_start_date", "contract_start_date_"]))
            end_date = parse_date_any(first_present(row, ["end_date", "contract_end_date", "contract_end_date_"]))

            first_seen = reg_date or start_date or date.today().isoformat()
            last_seen = reg_date or end_date or first_seen

            record: Dict[str, Any] = {
                "id": f"{CONFIG['SOURCE_NAME']}_rent_{listing_id}",
                "source": CONFIG["SOURCE_NAME"],
                "district": district,
                "price": price,
                "size_sqm": size_sqm,
                "bedrooms": bedrooms,
                "property_type": prop_norm or (raw_prop_type or "Unknown"),
                "latitude": None,
                "longitude": None,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "transaction_type": "rent",
                "rent_frequency": rent_frequency,
                "bathrooms": None,
                "furnishing": normalize_furnishing(to_str(first_present(row, ["furnishing", "furnished"]))),
                "building_name": to_str(first_present(row, ["building_name_en", "building_name", "building", "tower"])),
                "community": area,
                "subcommunity": to_str(first_present(row, ["subcommunity", "sub_community", "sub_area"])),
                "developer": to_str(first_present(row, ["developer", "developer_name"])),
                "floor": to_int(first_present(row, ["floor", "floor_no"])),
                "total_floors": to_int(first_present(row, ["total_floors", "floors", "building_levels"])),
                "parking_spaces": to_int(first_present(row, ["parking", "parking_spaces"])),
                "completion_status": normalize_completion_status(to_str(first_present(row, ["completion_status", "registration_type"]))),
                "year_built": to_int(first_present(row, ["year_built", "built_year", "construction_year"])),
                "service_charge_aed": None,
                "amenities": None,
                "listing_url": None,
                "agent_name": None,
                "agency_name": None,
                "photos_count": None,
                "title": "Ejari Rent Contract Record",
                "description_short": None,
                "verified_flag": None,
                "scraped_at": utc_now().isoformat(),
                "annual_amount_aed": annual_amount,
                "contract_amount_aed": contract_amount,
                "start_date": start_date,
                "end_date": end_date,
                "missing_size_flag": size_sqm is None,
                "raw_usage": usage,
                "raw_property_type": raw_prop_type,
                "raw_property_subtype": raw_subtype,
            }

            out.append(record)
            counts[district] += 1

            if len(out) % 500 == 0:
                logging.info("Rents collected so far: %d | per-district=%s", len(out), counts)

    finally:
        try:
            resp.raw.close()
        except Exception:
            pass

    logging.info("Finished rents ingestion: collected=%d scanned=%d per-district=%s", len(out), scanned, counts)
    return out


# =========================
# Post-processing / QA
# =========================
BASE_COLUMNS = [
    "id",
    "source",
    "district",
    "price",
    "size_sqm",
    "bedrooms",
    "property_type",
    "latitude",
    "longitude",
    "first_seen",
    "last_seen",
]

ENRICHED_COLUMNS = [
    "transaction_type",
    "rent_frequency",
    "bathrooms",
    "furnishing",
    "building_name",
    "community",
    "subcommunity",
    "developer",
    "floor",
    "total_floors",
    "parking_spaces",
    "completion_status",
    "year_built",
    "service_charge_aed",
    "amenities",
    "listing_url",
    "agent_name",
    "agency_name",
    "photos_count",
    "title",
    "description_short",
    "verified_flag",
    "scraped_at",
    "price_per_sqm",
    "days_on_market",
    "missing_size_flag",
    "raw_usage",
    "raw_property_type",
    "raw_property_subtype",
    "annual_amount_aed",
    "contract_amount_aed",
    "start_date",
    "end_date",
    # new (optional) - safe extra column
    "size_imputed_flag",
]


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    for c in BASE_COLUMNS + ENRICHED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    ordered = BASE_COLUMNS + [c for c in ENRICHED_COLUMNS if c not in BASE_COLUMNS]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["id"], keep="first")

    if CONFIG["DROP_UNKNOWN_DISTRICT"]:
        df = df[df["district"].isin(CONFIG["DISTRICTS"])].copy()

    # Coerce numeric early
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size_sqm"] = pd.to_numeric(df["size_sqm"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(0).astype(int)

    df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce").dt.date
    df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce").dt.date

    if "description_short" in df.columns:
        maxlen = int(CONFIG["DESCRIPTION_TRUNCATE_CHARS"])
        df["description_short"] = df["description_short"].astype("string").str.slice(0, maxlen)

    if CONFIG["DROP_IF_MISSING_BOTH_PRICE_AND_SIZE"]:
        missing_both = df["price"].isna() & df["size_sqm"].isna()
        df = df[~missing_both].copy()

    # 1) Fix lat/lon coverage (centroids + jitter)
    df = _fill_latlon_with_centroids_and_jitter(df)

    # 2) Fix size coverage (impute where possible)
    df = _impute_size_sqm(df)

    # price_per_sqm (compute AFTER size fixes)
    df["price_per_sqm"] = df["price"] / df["size_sqm"]
    df.loc[(df["size_sqm"].isna()) | (df["size_sqm"] <= 0), "price_per_sqm"] = pd.NA

    # days_on_market (kept but not meaningful for DLD transactions/contracts)
    first_dt = pd.to_datetime(df["first_seen"], errors="coerce")
    last_dt = pd.to_datetime(df["last_seen"], errors="coerce")
    dom = (last_dt - first_dt).dt.days
    df["days_on_market"] = dom.where(dom.notna(), pd.NA)

    if "rent_frequency" in df.columns:
        df["rent_frequency"] = df["rent_frequency"].replace({"": pd.NA})

    if "transaction_type" in df.columns:
        df["transaction_type"] = df["transaction_type"].astype("string").str.lower()

    # IMPORTANT: DLD data is not "listing exposure duration"
    df.loc[df["source"].astype(str).eq(CONFIG["SOURCE_NAME"]), "days_on_market"] = pd.NA

    return df


# =========================
# Main
# =========================
def main() -> None:
    setup_logging()

    for p in [CONFIG["LOCAL_TRANSACTIONS_PATH"], CONFIG["LOCAL_RENTS_PATH"]]:
        if not os.path.exists(p):
            raise SystemExit(f"Missing local file: {p} (put it there before running)")

    logging.info(
        "Config: districts=%s txn_types=%s max_rows_per=(district,txn)=%s | local_mode=ON",
        CONFIG["DISTRICTS"],
        CONFIG["TRANSACTION_TYPES"],
        CONFIG["MAX_ROWS_PER_DISTRICT_PER_TXN"],
    )

    sess = requests.Session()
    client = HttpClient(
        session=sess,
        timeout=(int(CONFIG["HTTP_CONNECT_TIMEOUT_SECONDS"]), int(CONFIG["HTTP_TIMEOUT_SECONDS"])),
        max_retries=int(CONFIG["HTTP_MAX_RETRIES"]),
        backoff_base=float(CONFIG["HTTP_BACKOFF_BASE_SECONDS"]),
        sleep_between=float(CONFIG["HTTP_SLEEP_BETWEEN_REQUESTS_SECONDS"]),
    )

    records: List[Dict[str, Any]] = []

    if "sale" in CONFIG["TRANSACTION_TYPES"]:
        try:
            records.extend(ingest_dld_transactions(client))
        except Exception as e:
            logging.error("Sale ingestion failed: %s", e)

    if "rent" in CONFIG["TRANSACTION_TYPES"]:
        try:
            records.extend(ingest_dld_rent_contracts(client))
        except Exception as e:
            logging.error("Rent ingestion failed: %s", e)

    if not records:
        raise SystemExit("No records collected. Check district mapping or local CSV contents.")

    df = pd.DataFrame.from_records(records)
    df = enforce_schema(df)
    df = clean_and_derive(df)

    # quick sanity logs
    logging.info("Output composition: %s", df["transaction_type"].value_counts(dropna=False).to_dict())
    logging.info("Output districts: %s", df["district"].value_counts(dropna=False).to_dict())
    logging.info("Missing size_sqm rate: %.3f", float(df["size_sqm"].isna().mean()))
    logging.info("Missing price rate: %.3f", float(df["price"].isna().mean()))
    logging.info("Missing latitude rate: %.3f", float(df["latitude"].isna().mean()))
    logging.info("Missing longitude rate: %.3f", float(df["longitude"].isna().mean()))
    if "size_imputed_flag" in df.columns:
        logging.info("Size imputed rate: %.3f", float(df["size_imputed_flag"].fillna(False).astype(bool).mean()))


    ensure_parent_dir(CONFIG["OUTPUT_CSV_PATH"])
    df.to_csv(CONFIG["OUTPUT_CSV_PATH"], index=False)

    if CONFIG.get("WRITE_PARQUET", False):
        ensure_parent_dir(CONFIG["OUTPUT_PARQUET_PATH"])
        try:
            df.to_parquet(CONFIG["OUTPUT_PARQUET_PATH"], index=False)
        except Exception as e:
            logging.warning("Parquet write failed. CSV still written. Error: %s", e)

    logging.info("Done. Rows=%d | CSV=%s", len(df), CONFIG["OUTPUT_CSV_PATH"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        sys.exit(130)
