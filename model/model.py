
import pandas as pd
import requests
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from xgboost import XGBClassifier


# ---------------- LOAD DATA ----------------
def load_data():

    url = "https://drive.google.com/uc?export=download&id=1uCSB6lS329wnOLrQCKHOj3ZxWYM6MGX-"

    df = pd.read_csv(io.StringIO(requests.get(url).content.decode("utf-8")))

    for col in ["listed_in", "director", "cast", "description"]:
        df[col] = df[col].fillna("")

    return df


# ======================================================
# RECOMMENDATION SYSTEM
# ======================================================
def get_recommendations(title):

    df = load_data()

    df["text"] = (
        df["listed_in"] + " " +
        df["director"] + " " +
        df["cast"] + " " +
        df["description"]
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    matrix = tfidf.fit_transform(df["text"])

    idx = pd.Series(
        df.index,
        index=df["title"]
    ).drop_duplicates()[title]

    similarity = cosine_similarity(
        matrix[idx],
        matrix
    ).flatten()

    ids = similarity.argsort()[-6:-1][::-1]

    return df.iloc[ids]["title"].tolist()


# ======================================================
# XGBOOST MODEL
# ======================================================
def train_xgboost():

    df = load_data()

    df["target"] = df["type"].map({
        "Movie": 0,
        "TV Show": 1
    })

    df["text"] = (
        df["description"] + " " +
        df["director"] + " " +
        df["cast"]
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=2500
    )

    X = tfidf.fit_transform(df["text"])
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

    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "cm": confusion_matrix(y_test, pred)
    }

