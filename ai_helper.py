
# ai_helper.py

import pandas as pd
import requests
import io


# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------
def load_data():

    url = "https://drive.google.com/uc?export=download&id=1uCSB6lS329wnOLrQCKHOj3ZxWYM6MGX-"

    df = pd.read_csv(
        io.StringIO(
            requests.get(url).content.decode("utf-8")
        )
    )

    for col in ["country", "listed_in", "duration", "type"]:
        df[col] = df[col].fillna("")

    return df


# ------------------------------------------------
# SMART AI ASSISTANT
# ------------------------------------------------
def ask_ai(question):

    df = load_data()

    q = question.lower()

    genres = [
        "romantic", "romance", "comedy", "thriller",
        "horror", "action", "drama", "family",
        "crime", "documentary", "anime"
    ]

    selected_genre = None

    for g in genres:
        if g in q:
            selected_genre = g
            break

    selected_country = None

    countries = ["india", "united states", "uk", "japan", "korea"]

    for c in countries:
        if c in q:
            selected_country = c
            break

    short_movie = False

    if "2 hour" in q or "under 2 hour" in q or "short" in q:
        short_movie = True

    # -----------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------
    if "watch" in q or "recommend" in q or "suggest" in q:

        result = df.copy()

        if selected_genre:
            result = result[
                result["listed_in"].str.contains(
                    selected_genre,
                    case=False,
                    na=False
                )
            ]

        if selected_country:
            result = result[
                result["country"].str.contains(
                    selected_country,
                    case=False,
                    na=False
                )
            ]

        if short_movie:
            result = result[
                result["duration"].str.contains(
                    "min",
                    na=False
                )
            ]

        titles = result["title"].dropna().head(5).tolist()

        if titles:
            return "Recommended for you: " + ", ".join(titles)

        else:
            return "No matching content found."

    # -----------------------------------------
    # POPULAR GENRE
    # -----------------------------------------
    elif "popular genre" in q or "best genre" in q:
        return f"Most popular genre is {df['listed_in'].mode()[0]}"

    # -----------------------------------------
    # INDIA POPULAR
    # -----------------------------------------
    elif "india" in q and "popular" in q:
        india = df[df["country"].str.contains("India", na=False)]

        return f"In India, popular category is {india['listed_in'].mode()[0]}"

    # -----------------------------------------
    # MOVIES VS TV SHOWS
    # -----------------------------------------
    elif "movie vs tv" in q:
        counts = df["type"].value_counts()

        return f"Movies: {counts['Movie']} | TV Shows: {counts['TV Show']}"

    # -----------------------------------------
    # TOTAL TITLES
    # -----------------------------------------
    elif "total" in q:
        return f"Total titles available: {len(df)}"

    # -----------------------------------------
    # GROWTH
    # -----------------------------------------
    elif "tv shows growing" in q:
        return "TV Shows are increasing due to binge-watch demand."

    # -----------------------------------------
    # DEFAULT
    # -----------------------------------------
    else:
        return 