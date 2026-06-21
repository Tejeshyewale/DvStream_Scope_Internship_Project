"""
utils/vector_store.py

Semantic (RAG) search over the StreamScope dataset, replacing the
old substring/keyword matching in ai_helper.build_context().

How it works:
  1. Every title's combined text (genre + director + cast +
     description) is converted into a numeric vector ("embedding")
     using a small local sentence-transformers model. Similar
     meaning -> similar vector, even if the exact words differ.
  2. All vectors are stored in a FAISS index, which can quickly
     find the closest vectors to a query.
  3. When a user asks a question, we embed the question the same
     way and search the index for the most semantically similar
     titles -- this is the "retrieval" half of Retrieval-Augmented
     Generation (RAG).
  4. Those retrieved titles are handed to the LLM as grounding
     context (done in ai_helper.py), so it answers using real
     catalog data instead of guessing.

Everything here is cached for the Streamlit session so the model
is downloaded/loaded and the index is built only once.
"""

import numpy as np

from utils.data_loader import load_data

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free, local


def _try_streamlit_cache_resource():
    try:
        import streamlit as st
        return st.cache_resource(show_spinner="Building semantic search index...")
    except Exception:
        def _noop(fn):
            return fn
        return _noop


_cache_resource = _try_streamlit_cache_resource()


@_cache_resource
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@_cache_resource
def _build_vector_index():
    """
    Builds (once, cached) a FAISS index over every title's combined
    text field. Returns (df, faiss_index).
    """
    import faiss

    df = load_data().reset_index(drop=True)
    model = _get_embedding_model()

    texts = df["combined_text"].fillna("").tolist()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return df, index


def semantic_search(query: str, top_k: int = 15):
    """
    Returns a DataFrame slice of the top_k titles most semantically
    similar to the query. This is the core retrieval step of RAG --
    it understands meaning, not just exact keyword overlap, so
    "scary movies" can match titles tagged "Horror" even without
    the word "scary" appearing anywhere in the dataset.
    """
    df, index = _build_vector_index()
    model = _get_embedding_model()

    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_vec, top_k)

    valid = [i for i in indices[0] if i != -1]
    results = df.iloc[valid].copy()
    results["similarity"] = scores[0][: len(valid)]

    return results
