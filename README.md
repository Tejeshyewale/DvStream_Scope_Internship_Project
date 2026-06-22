# StreamScope AI

A Netflix-style content intelligence platform combining analytics, a content-based recommendation engine, and an agentic AI assistant — built on a real-world streaming catalog dataset (8,800+ titles).

Built during an internship project, then extended with retrieval-augmented generation, agentic tool-calling, and a full model-evaluation suite.

### Here you can checkout the live demo : https://streamscope-tejesh-417.streamlit.app/

## Features

- **Dashboard** — catalog KPIs, content distribution, top countries
- **Discover** — TF-IDF + cosine-similarity content recommender
- **Analytics** — growth trends, genre breakdown, country treemap
- **AI Lab** — conversational assistant with two modes:
  - *RAG mode*: semantic vector search grounds every answer in real catalog data
  - *Agentic mode*: the LLM decides which tools to call (semantic search, year filter, statistics) based on the question
  - Multi-turn conversation memory, response caching, and a live tool-trace so you can see what the agent did
- **Model Lab** — XGBoost classifier (Movie vs TV Show) with accuracy/precision/recall/F1, confusion matrix, ROC curve, and feature importance

## Architecture

```
                         ┌─────────────┐
                         │   app.py    │   Streamlit UI (5 pages)
                         └──────┬──────┘
              ┌─────────────────┼─────────────────┐
              │                 │                 │
       ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
       │  model.py   │   │ ai_helper.py│   │ data_loader │
       │ TF-IDF      │   │ Groq LLM    │   │ Shared CSV  │
       │ recommender │   │ RAG +       │   │ loader,     │
       │ XGBoost     │   │ agentic     │   │ cached      │
       │ classifier  │   │ chat        │   │             │
       └─────────────┘   └──────┬──────┘   └─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
       ┌──────▼──────┐   ┌───────▼──────┐   ┌───────▼──────┐
       │vector_store │   │  ai_tools    │   │    cache     │
       │ Sentence-   │   │ semantic_    │   │ Context-aware│
       │ Transformers│   │ search,      │   │ response     │
       │ + FAISS     │   │ filter_by_   │   │ caching      │
       │             │   │ year, stats  │   │              │
       └─────────────┘   └──────────────┘   └──────────────┘
```

**Request flow (Agentic mode):** user question → Groq LLM picks tool(s) → tool runs against the dataset → results returned to the LLM → grounded JSON answer → rendered as chat reply + recommendation cards in the UI.

## Tech stack

| Layer | Tools |
|---|---|
| UI | Streamlit, Plotly |
| LLM | Groq (Llama 3.3 70B), tool-calling / function-calling |
| RAG | sentence-transformers (`all-MiniLM-L6-v2`), FAISS |
| ML | scikit-learn (TF-IDF, cosine similarity), XGBoost |
| Data | pandas, numpy |

## Project structure

```
DvStream_Scope_Internship_Project/
├── app.py                  # Streamlit UI — all 5 pages
├── ai_helper.py             # RAG + agentic chat logic (Groq)
├── requirements.txt
├── model/
│   ├── __init__.py
│   └── model.py              # TF-IDF recommender + XGBoost classifier
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # Shared, cached dataset loader
│   ├── vector_store.py         # Embeddings + FAISS semantic search
│   ├── ai_tools.py             # Tool definitions for agentic mode
│   └── cache.py                 # AI response caching
├── archive/
│   └── netflix_dashboard_extended.py   # Earlier draft, kept for reference
└── .streamlit/
    └── secrets.toml            # GROQ_API_KEY (not committed)
```

## Setup

```bash
git clone https://github.com/Tejeshyewale/DvStream_Scope_Internship_Project.git
cd DvStream_Scope_Internship_Project
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

```bash
streamlit run app.py
```

## Notable engineering decisions

- **Single shared data loader** — the dataset is fetched once per session and reused across the recommender, classifier, and AI assistant, instead of three separate network calls.
- **Context-aware caching** — identical questions only hit the Groq API once; the cache key includes recent chat history so follow-up questions in a different conversation don't return a stale answer.
- **Graceful degradation** — missing titles, duplicate dataset rows, and unexpected category values are handled without crashing the UI.

## License

See [LICENSE](LICENSE).
