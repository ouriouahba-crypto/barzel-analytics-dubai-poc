# src/db/db.py
from __future__ import annotations

import os
from sqlalchemy import create_engine


def get_database_url() -> str:
    """
    Resolution order (clean & production-safe):
    1) Streamlit Cloud secrets injected as env vars
    2) OS environment variable
    3) Local .env (development fallback)
    """

    # 1) Streamlit Cloud injects secrets as env vars
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    # 2) Local .env fallback (dev only)
    try:
        from dotenv import load_dotenv

        load_dotenv()
        url = os.getenv("DATABASE_URL")
        if url:
            return url
    except Exception:
        pass

    raise ValueError(
        "DATABASE_URL is missing.\n"
        "Set it in Streamlit Cloud: Manage app → Settings → Secrets\n"
        "or export DATABASE_URL locally / put it in a .env file."
    )


def get_engine():
    url = get_database_url()
    return create_engine(url, pool_pre_ping=True)
