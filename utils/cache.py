"""
utils/cache.py

Caches AI responses so repeated/identical questions don't trigger a
fresh Groq API call every time -- faster for the user and cheaper
on API usage.

The cache key includes a bit of recent chat history (not just the
raw question), because the same question can have a different
correct answer depending on conversation context (e.g. "which one
is the oldest?" means something different after "romantic movies"
vs after "horror movies"). Two near-identical conversations should
share a cache entry; two different conversations should not.

Storage: Streamlit's session_state, so the cache is per-user-session
(no cross-user data leakage) and resets naturally when the session
ends. Falls back to a plain module-level dict outside Streamlit
(e.g. for tests).
"""

import hashlib
import json

_FALLBACK_STORE = {}

MAX_CACHE_ENTRIES = 200


def _make_cache_key(question: str, chat_history: list, mode: str) -> str:
    """
    Builds a stable cache key from the question, the last couple of
    chat turns (for context-sensitivity), and which code path was
    used (agentic vs plain RAG), since the two can produce different
    answers for the same question.
    """
    recent = chat_history[-4:] if chat_history else []
    key_payload = {
        "question": question.strip().lower(),
        "recent": recent,
        "mode": mode,
    }
    raw = json.dumps(key_payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_store():
    try:
        import streamlit as st

        if "ai_response_cache" not in st.session_state:
            st.session_state.ai_response_cache = {}
        return st.session_state.ai_response_cache

    except Exception:
        return _FALLBACK_STORE


def get_cached_response(question: str, chat_history: list, mode: str):
    """
    Returns the cached response dict if this exact question (in this
    exact recent context, for this mode) was already answered, else
    None.
    """
    store = _get_store()
    key = _make_cache_key(question, chat_history, mode)
    return store.get(key)


def set_cached_response(question: str, chat_history: list, mode: str, response: dict):
    store = _get_store()
    key = _make_cache_key(question, chat_history, mode)

    # Simple bound so the cache doesn't grow unbounded in a very
    # long-running session.
    if len(store) >= MAX_CACHE_ENTRIES:
        oldest_key = next(iter(store))
        store.pop(oldest_key, None)

    store[key] = response


def clear_cache():
    store = _get_store()
    store.clear()
