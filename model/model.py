
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

from xgboost import XGBClassifier

from utils.data_loader import load_data


def _try_streamlit_cache_resource():
    """
    Returns st.cache_resource if Streamlit is available, otherwise a
    no-op decorator. cache_resource is used (not cache_data) because
    we're caching a fitted vectorizer/matrix object, not plain data.
    """
    try:
        import streamlit as st
        return st.cache_resource(show_spinner=False)
    except Exception:
        def _noop(fn):
            return fn
        return _noop


_cache_resource = _try_streamlit_cache_resource()


# ======================================================
# RECOMMENDATION SYSTEM (TF-IDF + cosine similarity)
# ======================================================
@_cache_resource
def _build_recommender():
    """
    Builds the TF-IDF matrix once and caches it for the session,
    instead of recomputing it on every recommendation request.
    Returns (df, tfidf_matrix, title_to_index_map).
    """
    df = load_data().reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(df["combined_text"])

    # Build an explicit title -> first-matching-index map. Using a
    # dict instead of pandas' .drop_duplicates()[title] avoids a
    # crash when duplicate titles exist (previously could return a
    # Series instead of a scalar and break indexing downstream).
    title_to_index = {}
    for i, t in enumerate(df["title"]):
        if t not in title_to_index:
            title_to_index[t] = i

    return df, matrix, title_to_index


def get_recommendations(title, top_n=5):
    """
    Returns up to top_n recommended titles similar to the given title.
    Returns an empty list (instead of raising) if the title isn't
    found, so the UI can handle it gracefully.
    """
    df, matrix, title_to_index = _build_recommender()

    idx = title_to_index.get(title)
    if idx is None:
        return []

    similarity = cosine_similarity(matrix[idx], matrix).flatten()

    # Exclude the title itself, then take the top N matches.
    ranked = similarity.argsort()[::-1]
    ranked = [i for i in ranked if i != idx][:top_n]

    return df.iloc[ranked]["title"].tolist()


# ======================================================
# XGBOOST MODEL (Movie vs TV Show classifier)
# ======================================================
@_cache_resource
def train_xgboost():
    df = load_data()

    # Guard against unexpected "type" values (NaN, typos, extra
    # categories) which would otherwise crash the binary metrics
    # (precision_score/f1_score assume exactly two classes).
    df = df[df["type"].isin(["Movie", "TV Show"])].copy()

    if df["type"].nunique() < 2:
        raise ValueError(
            "Dataset must contain both 'Movie' and 'TV Show' rows "
            "to train the classifier."
        )

    df["target"] = df["type"].map({"Movie": 0, "TV Show": 1})

    tfidf = TfidfVectorizer(stop_words="english", max_features=2500)
    X = tfidf.fit_transform(df["combined_text"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, pred_proba)
    auc = roc_auc_score(y_test, pred_proba)

    # Top features by XGBoost's gain-based importance, mapped back
    # to their actual TF-IDF terms so the chart is human-readable
    # (e.g. "season", "documentary") instead of raw feature indices.
    feature_names = tfidf.get_feature_names_out()
    importances = model.feature_importances_
    top_n = 15
    top_idx = importances.argsort()[::-1][:top_n]
    feature_importance = [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in top_idx
    ]

    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "cm": confusion_matrix(y_test, pred),
        "model": model,
        "y_test": y_test,
        "pred": pred,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "feature_importance": feature_importance,
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
    }
