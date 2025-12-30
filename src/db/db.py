# src/db/db.py
from __future__ import annotations

import os
from sqlalchemy import create_engine


def get_database_url() -> str:
    """
    Priority:
    1) Streamlit Cloud secrets (st.secrets["DATABASE_URL"])
    2) Environment variable (DATABASE_URL)
    3) Local .env (via python-dotenv) - optional fallback
    """
    # 1) Streamlit secrets (Cloud)
    try:
        import streamlit as st  # local import to avoid issues in non-streamlit contexts

        if "DATABASE_URL" in st.secrets:
            return str(st.secrets["DATABASE_URL"])
    except Exception:
        # If streamlit isn't available or secrets not configured, ignore
        pass

    # 2) OS env var
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    # 3) Local .env fallback (optional)
    try:
        from dotenv import load_dotenv

        load_dotenv()
        url = os.getenv("DATABASE_URL")
        if url:
            return url
    except Exception:
        pass

    raise ValueError(
        "DATABASE_URL is missing. "
        "Set it in Streamlit Cloud: Manage app → Settings → Secrets as DATABASE_URL, "
        "or export DATABASE_URL locally / put it in .env."
    )


def get_engine():
    url = get_database_url()

    # SQLAlchemy accepts postgresql://... and postgresql+psycopg2://...
    return create_engine(url, pool_pre_ping=True)
