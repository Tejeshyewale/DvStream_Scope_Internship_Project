"""
utils/ai_tools.py

Defines the "tools" (functions) that the Groq LLM can choose to call
on its own, turning ai_helper from a fixed RAG pipeline into a real
agent. Instead of us always doing semantic search and handing the
LLM whatever we found, the LLM now decides which of these tools (if
any, and how many) it needs to answer a given question:

  - semantic_search_tool : find titles by meaning/genre/mood
  - filter_by_year_tool  : find titles released in/after/before a year
  - get_statistics_tool  : counts and breakdowns (totals, genres, countries)

Each tool has a JSON-schema "definition" (what Groq sees) and a
matching Python function (what actually runs when Groq calls it).
"""

import json

from utils.data_loader import load_data
from utils.vector_store import semantic_search


# ======================================================
# TOOL 1: Semantic search (the RAG retrieval from Step 2)
# ======================================================
def semantic_search_tool(query: str, top_k: int = 10) -> str:
    results = semantic_search(query, top_k=top_k)

    cols = [c for c in ["title", "type", "listed_in", "country", "release_year"] if c in results.columns]
    records = results[cols].to_dict(orient="records")

    return json.dumps({"results": records}, default=str)


# ======================================================
# TOOL 2: Filter by year / year range
# ======================================================
def filter_by_year_tool(year_from: int = None, year_to: int = None, limit: int = 15) -> str:
    df = load_data()

    if "release_year" not in df.columns:
        return json.dumps({"error": "release_year column not available"})

    subset = df
    if year_from is not None:
        subset = subset[subset["release_year"] >= year_from]
    if year_to is not None:
        subset = subset[subset["release_year"] <= year_to]

    cols = [c for c in ["title", "type", "listed_in", "country", "release_year"] if c in subset.columns]
    records = subset[cols].head(limit).to_dict(orient="records")

    return json.dumps({"count_matched": int(len(subset)), "results": records}, default=str)


# ======================================================
# TOOL 3: Dataset statistics
# ======================================================
def get_statistics_tool() -> str:
    df = load_data()

    stats = {
        "total_titles": int(len(df)),
        "total_movies": int((df["type"] == "Movie").sum()),
        "total_tv_shows": int((df["type"] == "TV Show").sum()),
        "unique_countries": int(df["country"].nunique()),
    }

    if "listed_in" in df.columns:
        genre_counts = (
            df["listed_in"]
            .str.split(",")
            .explode()
            .str.strip()
            .value_counts()
            .head(5)
        )
        stats["top_genres"] = genre_counts.to_dict()

    if "country" in df.columns:
        country_counts = (
            df["country"]
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", None)
            .dropna()
            .value_counts()
            .head(5)
        )
        stats["top_countries"] = country_counts.to_dict()

    return json.dumps(stats, default=str)


# ======================================================
# TOOL REGISTRY -- definitions Groq sees + functions we run
# ======================================================
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search_tool",
            "description": (
                "Search the content catalog by meaning, genre, mood, or theme "
                "(e.g. 'scary movies', 'feel-good family shows', 'movies about "
                "time travel'). Use this whenever the user wants recommendations "
                "or is browsing by topic/genre/vibe."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing what kind of content to find",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many results to return (default 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_year_tool",
            "description": (
                "Filter the catalog by release year or year range "
                "(e.g. 'released after 2020', 'movies from 2015 to 2018'). "
                "Use this whenever the question mentions a specific year, "
                "decade, or date range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year_from": {
                        "type": "integer",
                        "description": "Earliest release year to include (inclusive). Omit if not specified.",
                    },
                    "year_to": {
                        "type": "integer",
                        "description": "Latest release year to include (inclusive). Omit if not specified.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return (default 15)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_statistics_tool",
            "description": (
                "Get aggregate statistics about the catalog: total number of "
                "titles, movie vs TV show counts, number of countries, top "
                "genres, and top countries. Use this for questions like "
                "'how many movies are there' or 'what are the top genres'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "semantic_search_tool": semantic_search_tool,
    "filter_by_year_tool": filter_by_year_tool,
    "get_statistics_tool": get_statistics_tool,
}
