
# ai_helper.py

import os
import json

from groq import Groq

from utils.data_loader import load_data
from utils.vector_store import semantic_search
from utils.ai_tools import TOOL_DEFINITIONS, AVAILABLE_FUNCTIONS
from utils.cache import get_cached_response, set_cached_response

MODEL_NAME = "llama-3.3-70b-versatile"


# ------------------------------------------------
# GROQ CLIENT
# ------------------------------------------------
def get_groq_client():
    """
    Reads the API key from Streamlit secrets first (recommended for
    Streamlit Cloud), falling back to an environment variable for
    local development.
    """
    api_key = None

    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Add it to .streamlit/secrets.toml "
            "or set it as an environment variable."
        )

    return Groq(api_key=api_key)


# ------------------------------------------------
# BUILD DATASET CONTEXT FOR THE MODEL (RAG)
# ------------------------------------------------
def build_context(df, question, top_k=15):
    """
    Retrieves the titles most semantically relevant to the question
    using vector search (RAG), instead of plain keyword/substring
    matching. This understands meaning -- e.g. "scary movies" can
    surface titles tagged "Horror" even though the word "scary"
    never appears in the dataset, because their embeddings are
    close in vector space.
    """
    results = semantic_search(question, top_k=top_k)

    cols = [c for c in ["title", "type", "listed_in", "country", "release_year"] if c in df.columns]

    lines = [
        f"- {getattr(r, 'title', '')} ({getattr(r, 'type', '')}, {getattr(r, 'release_year', '')}) "
        f"| Genre: {getattr(r, 'listed_in', '')} | Country: {getattr(r, 'country', '')} "
        f"| Relevance: {getattr(r, 'similarity', 0):.2f}"
        for r in results[cols + ["similarity"]].itertuples()
    ]

    stats = (
        f"Dataset stats: {len(df)} total titles, "
        f"{(df['type'] == 'Movie').sum()} movies, "
        f"{(df['type'] == 'TV Show').sum()} TV shows, "
        f"{df['country'].nunique()} countries."
    )

    return (
        stats
        + "\n\nSemantically relevant titles for this question "
        + "(ranked by relevance, 1.0 = perfect match):\n"
        + "\n".join(lines)
    )


def _build_system_prompt(context, structured=False):
    base = (
        "You are StreamScope AI, a helpful content assistant for the "
        "StreamScope app. Answer the user's question using ONLY the "
        "dataset context provided below. Recommend real titles from "
        "the context when asked for recommendations. Be concise and "
        "friendly. If the context doesn't have a good match, say so "
        "honestly instead of making titles up.\n\n"
        "You may be in the middle of a multi-turn conversation -- use "
        "the earlier messages to understand follow-up questions (e.g. "
        "\"which of those is the newest?\" refers back to titles you "
        "already mentioned).\n\n"
        f"{context}"
    )

    if structured:
        base += (
            "\n\nRespond with STRICT JSON only, no markdown fences, in "
            "exactly this shape:\n"
            '{"reply": "<2-3 sentence conversational answer>", '
            '"recommendations": [{"title": "<exact title from context>", '
            '"year": "<release_year>", "genre": "<short genre tag>", '
            '"type": "Movie or TV Show"}]}\n'
            "Include up to 5 recommendations only if the question asks "
            "for suggestions/recommendations/watch ideas. Otherwise "
            'return an empty list for "recommendations".'
        )

    return base


# ------------------------------------------------
# STRUCTURED RESPONSE (for card/poster UI)
# ------------------------------------------------
def ask_ai_structured(question, chat_history=None):
    """
    Returns a dict: {"reply": str, "recommendations": [ {title, year,
    genre, type}, ... ]}. Used to render movie/show cards in the UI.

    chat_history (optional): list of {"role": "user"/"assistant",
    "content": str} dicts from earlier turns in the conversation.
    When provided, it's used two ways:
      1. Appended to the messages sent to Groq, so follow-up
         questions ("which of these is the oldest?") are understood
         in context.
      2. Folded into the RAG retrieval query, so vector search also
         stays on-topic for follow-ups that don't repeat the
         original subject (e.g. "romantic movies" -> "the oldest
         one?" should still retrieve romance titles, not random ones).
    """
    chat_history = chat_history or []

    cached = get_cached_response(question, chat_history, mode="structured")
    if cached is not None:
        return {**cached, "from_cache": True}

    df = load_data()

    # Build a retrieval query that includes recent context, so RAG
    # search understands follow-ups, not just the latest message.
    recent_user_turns = [
        m["content"] for m in chat_history[-4:] if m.get("role") == "user"
    ]
    retrieval_query = " ".join(recent_user_turns + [question])

    context = build_context(df, retrieval_query)
    system_prompt = _build_system_prompt(context, structured=True)

    messages = [{"role": "system", "content": system_prompt}]
    # Include prior turns (capped) so the model has conversational
    # context without the prompt growing unbounded.
    messages.extend(chat_history[-8:])
    messages.append({"role": "user", "content": question})

    try:
        client = get_groq_client()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4,
            max_completion_tokens=500,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        result = {
            "reply": data.get("reply", ""),
            "recommendations": data.get("recommendations", []) or [],
        }

        set_cached_response(question, chat_history, mode="structured", response=result)

        return {**result, "from_cache": False}

    except Exception as e:
        return {
            "reply": f"AI service error: {e}",
            "recommendations": [],
            "from_cache": False,
        }


# ------------------------------------------------
# STREAMING RESPONSE (plain text, for typing effect)
# ------------------------------------------------
def ask_ai_stream(question):
    """
    Generator that yields response text chunks as they arrive from
    Groq, so the UI can render a typing effect with st.write_stream.
    """
    df = load_data()
    context = build_context(df, question)
    system_prompt = _build_system_prompt(context, structured=False)

    try:
        client = get_groq_client()

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.4,
            max_completion_tokens=300,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as e:
        yield f"AI service error: {e}"


# ------------------------------------------------
# SIMPLE NON-STREAMING (kept for backward compatibility)
# ------------------------------------------------
def ask_ai(question):
    df = load_data()
    context = build_context(df, question)
    system_prompt = _build_system_prompt(context, structured=False)

    try:
        client = get_groq_client()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.4,
            max_completion_tokens=300,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI service error: {e}"


# ======================================================
# AGENTIC TOOL-CALLING (Step 4)
# ======================================================
def ask_ai_agentic(question, chat_history=None, max_tool_iterations=4):
    """
    Unlike ask_ai_structured() (which always runs semantic search and
    hands the LLM whatever it finds), this version gives the LLM three
    tools -- semantic_search_tool, filter_by_year_tool, and
    get_statistics_tool -- and lets it decide which ones to call, in
    what order, and with what arguments, based on the actual question.

    Returns a dict shaped like ask_ai_structured()'s output:
    {"reply": str, "recommendations": [...], "tool_trace": [...]}
    "tool_trace" is included so the UI can optionally show the user
    which tools the agent decided to use (nice for demos).
    """
    chat_history = chat_history or []

    cached = get_cached_response(question, chat_history, mode="agentic")
    if cached is not None:
        return {**cached, "from_cache": True}

    system_prompt = (
        "You are StreamScope AI, an agentic content assistant. You have "
        "access to tools that query a real content catalog -- use them "
        "to ground every factual claim or recommendation in real data. "
        "Never invent titles, years, or statistics; always call a tool "
        "first if the question needs catalog data.\n\n"
        "Guidance:\n"
        "- For recommendations / genre / mood / theme questions, use "
        "semantic_search_tool.\n"
        "- For questions mentioning a specific year or year range, use "
        "filter_by_year_tool (you can combine it with semantic_search_tool "
        "if both a topic and a year are mentioned).\n"
        "- For counts, totals, or 'top genres/countries' questions, use "
        "get_statistics_tool.\n"
        "- You may call more than one tool, or the same tool more than "
        "once, before answering.\n\n"
        "Once you have enough information, respond with STRICT JSON only "
        "(no markdown fences), in exactly this shape:\n"
        '{"reply": "<2-3 sentence conversational answer>", '
        '"recommendations": [{"title": "<exact title from tool results>", '
        '"year": "<release_year>", "genre": "<short genre tag>", '
        '"type": "Movie or TV Show"}]}\n'
        "Include up to 5 recommendations only if relevant titles were "
        'found via tools; otherwise return an empty list for "recommendations".'
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-8:])
    messages.append({"role": "user", "content": question})

    tool_trace = []

    try:
        client = get_groq_client()

        for _ in range(max_tool_iterations):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.3,
                max_completion_tokens=600,
            )

            choice = response.choices[0]
            response_message = choice.message
            messages.append(response_message)

            tool_calls = response_message.tool_calls or []
            if not tool_calls:
                # The model is done calling tools and has (hopefully)
                # produced its final answer in response_message.content.
                break

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    function_args = {}

                function_to_call = AVAILABLE_FUNCTIONS.get(function_name)

                if function_to_call is None:
                    function_response = json.dumps({"error": f"Unknown tool: {function_name}"})
                else:
                    try:
                        function_response = function_to_call(**function_args)
                    except Exception as tool_error:
                        function_response = json.dumps({"error": str(tool_error)})

                tool_trace.append({"tool": function_name, "args": function_args})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response,
                })

        # Force a final, tool-free, JSON-formatted answer using
        # everything gathered so far.
        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=500,
            response_format={"type": "json_object"},
        )

        raw = final_response.choices[0].message.content
        data = json.loads(raw)

        result = {
            "reply": data.get("reply", ""),
            "recommendations": data.get("recommendations", []) or [],
            "tool_trace": tool_trace,
        }

        set_cached_response(question, chat_history, mode="agentic", response=result)

        return {**result, "from_cache": False}

    except Exception as e:
        return {
            "reply": f"AI service error: {e}",
            "recommendations": [],
            "tool_trace": tool_trace,
            "from_cache": False,
        }
