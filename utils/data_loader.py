"""
utils/data_loader.py

Single source of truth for loading the StreamScope content dataset.
Previously, app.py, model/model.py, and ai_helper.py each had their
own copy of this CSV-loading logic -- meaning the dataset was
downloaded from Google Drive multiple times per session and any
fix had to be applied in three places. This module centralizes it.

Uses Streamlit's cache when available (so the CSV is fetched once
per session) and falls back to a plain in-memory cache for non-
Streamlit contexts (e.g. unit tests, scripts).
"""

import io
import pandas as pd
import requests

DATA_URL = "https://drive.google.com/uc?export=download&id=1uCSB6lS329wnOLrQCKHOj3ZxWYM6MGX-"

_FALLBACK_CACHE = {}


def _fetch_raw() -> pd.DataFrame:
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.content.decode("utf-8")))


def _load_and_clean() -> pd.DataFrame:
    df = _fetch_raw()

    text_cols = ["country", "listed_in", "duration", "type", "director", "cast", "description"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    if "date_added" in df.columns:
        df["year_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.year

    # Combined text field reused by both the TF-IDF recommender and
    # the XGBoost classifier, so we only build it once.
    df["combined_text"] = (
        df.get("listed_in", "") + " " +
        df.get("director", "") + " " +
        df.get("cast", "") + " " +
        df.get("description", "")
    ).str.strip()

    return df


def load_data() -> pd.DataFrame:
    """
    Returns the cleaned StreamScope dataset as a DataFrame.
    Cached for the lifetime of the Streamlit session when running
    inside Streamlit; cached in-process otherwise.
    """
    try:
        import streamlit as st

        @st.cache_data(show_spinner=False)
        def _cached_load():
            return _load_and_clean()

        return _cached_load()

    except Exception:
        # Not running inside Streamlit (e.g. a script or test) --
        # use a simple module-level cache instead.
        if "df" not in _FALLBACK_CACHE:
            _FALLBACK_CACHE["df"] = _load_and_clean()
        return _FALLBACK_CACHE["df"].copy()
