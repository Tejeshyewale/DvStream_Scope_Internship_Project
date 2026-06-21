import streamlit as st, pandas as pd, plotly.express as px
from model import train_xgboost, get_recommendations
from ai_helper import ask_ai_structured, ask_ai_agentic
from utils.data_loader import load_data as load_dataset

st.set_page_config(page_title='StreamScope AI', layout='wide', page_icon='🎬')

st.markdown("""
<style>
.stApp{
    background:linear-gradient(180deg,#050505,#000);
    color:#fff;
}
.block-container{
    max-width:1450px;
    padding-top:.4rem;
}
header{
    visibility:hidden;
}
.stButton>button{
    background:#141414!important;
    color:#fff!important;
    border:1px solid #E50914!important;
    border-radius:14px!important;
    box-shadow:0 0 12px rgba(229,9,20,.35)!important;
    font-weight:700!important;
}
.stButton>button:hover{
    background:#E50914!important;
    color:#fff!important;
}
.card{
    background:#101010;
    border:1px solid #2b2b2b;
    border-radius:18px;
    padding:16px;
    box-shadow:0 0 14px rgba(229,9,20,.12);
}
.kpi{
    font-size:32px;
    font-weight:800;
}
.muted{
    color:#bbb;
}
.chatfab{
    position:fixed;
    right:22px;
    bottom:22px;
    width:64px;
    height:64px;
    border-radius:50%;
    background:#E50914;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:30px;
    box-shadow:0 0 18px rgba(229,9,20,.55);
}
.brand-title{
    font-size:42px;
    font-weight:900;
    letter-spacing:1px;
    margin-bottom:4px;
}
.brand-sub{
    color:#999;
    font-size:14px;
    margin-bottom:18px;
}
.ai-reply{
    background:#101010;
    border:1px solid #2b2b2b;
    border-left:4px solid #E50914;
    border-radius:14px;
    padding:18px 20px;
    margin:14px 0 22px 0;
    line-height:1.55;
    box-shadow:0 0 14px rgba(229,9,20,.10);
}
.poster-card{
    background:#101010;
    border:1px solid #2b2b2b;
    border-radius:16px;
    padding:14px;
    height:100%;
    transition:transform .15s ease, box-shadow .15s ease;
    box-shadow:0 0 10px rgba(229,9,20,.08);
}
.poster-card:hover{
    transform:translateY(-4px);
    box-shadow:0 0 20px rgba(229,9,20,.30);
    border-color:#E50914;
}
.poster-badge{
    display:inline-block;
    font-size:11px;
    font-weight:700;
    color:#E50914;
    background:rgba(229,9,20,.12);
    border:1px solid rgba(229,9,20,.4);
    border-radius:20px;
    padding:2px 10px;
    margin-bottom:8px;
}
.poster-title{
    font-size:16px;
    font-weight:700;
    margin-bottom:4px;
    min-height:42px;
}
.poster-meta{
    color:#999;
    font-size:12.5px;
}
</style>
""", unsafe_allow_html=True)


df = load_dataset().copy()

# Dashboard display prefers "Unknown" labels over blank strings for
# missing country/genre values (shared loader uses "" so other
# modules can do clean substring matching).
df["country"] = df["country"].replace("", "Unknown")
df["listed_in"] = df["listed_in"].replace("", "Unknown")

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

nav = st.columns(5)

for c, p in zip(
    nav,
    ["🏠 Dashboard", "🎬 Discover", "📊 Analytics", "🤖 AI Lab", "📈 Model Lab"]
):
    if c.button(p, use_container_width=True, key=p):
        st.session_state.page = p

page = st.session_state.page

st.markdown(
    """
    <div class='brand-title'><span style='color:#E50914'>STREAM</span>SCOPE 🤖 AI</div>
    <div class='brand-sub'>Your AI-powered content explorer, built on real catalog data</div>
    """,
    unsafe_allow_html=True
)

# ======================================================
# DASHBOARD
# ======================================================
if page == "🏠 Dashboard" or page == "Dashboard":

    a, b, c, d = st.columns(4)

    vals = [
        len(df),
        (df["type"] == "Movie").sum(),
        (df["type"] == "TV Show").sum(),
        df["country"].nunique()
    ]

    labs = [
        "Total Titles",
        "Movies",
        "TV Shows",
        "Countries"
    ]

    for col, v, l in zip([a, b, c, d], vals, labs):
        col.markdown(
            f"""
            <div class='card'>
                <div class='muted'>{l}</div>
                <div class='kpi'>{v}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    x, y = st.columns(2)

    fig1 = px.pie(
        df,
        names="type",
        hole=.68,
        color_discrete_sequence=["#E50914", "#555"]
    )

    fig1.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title="Content Distribution"
    )

    x.plotly_chart(fig1, use_container_width=True)

    top = df["country"].value_counts().head(8)

    fig2 = px.bar(
        x=top.values,
        y=top.index,
        orientation="h",
        color=top.values,
        color_continuous_scale="Reds",
        title="Top Countries"
    )

    fig2.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    y.plotly_chart(fig2, use_container_width=True)

# ======================================================
# DISCOVER
# ======================================================
elif "Discover" in page:

    st.subheader("🎬 Recommendation Engine")

    t = st.selectbox(
        "Choose Title",
        sorted(df["title"].dropna().unique())
    )

    if st.button(
        "Generate Recommendations",
        use_container_width=True
    ):

        rec = get_recommendations(t)

        if not rec:
            st.warning("No recommendations found for this title yet. Try another one!")
        else:
            cols = st.columns(len(rec[:5]))

            for i, r in enumerate(rec[:5]):
                cols[i].markdown(
                    f"""
                    <div class='card'>
                        <b>{r}</b><br>
                        <span class='muted'>Recommended</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ======================================================
# ANALYTICS (3 CHARTS)
# ======================================================
elif "Analytics" in page:

    st.subheader("📊 Premium Analytics")

    # Row 1
    c1, c2 = st.columns(2)

    # Chart 1: Growth Trend
    growth = df.groupby("year_added").size().reset_index(name="count").dropna()

    fig1 = px.line(
        growth,
        x="year_added",
        y="count",
        markers=True,
        title="Content Growth Over Time",
        color_discrete_sequence=["#E50914"]
    )

    fig1.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    c1.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Top Genres
    genres = df["listed_in"].value_counts().head(10)

    fig2 = px.bar(
        x=genres.values,
        y=genres.index,
        orientation="h",
        title="Top Genres",
        color=genres.values,
        color_continuous_scale="Reds"
    )

    fig2.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    c2.plotly_chart(fig2, use_container_width=True)

    # Row 2 Full Width
    country = df["country"].value_counts().head(12).reset_index()
    country.columns = ["country", "count"]

    fig3 = px.treemap(
        country,
        path=["country"],
        values="count",
        title="Country Content Share",
        color="count",
        color_continuous_scale="Reds"
    )

    fig3.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ======================================================
# AI LAB (StreamScope AI Assistant)
# ======================================================
elif "AI Lab" in page:

    st.subheader("🤖 StreamScope AI Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role", "content"}
    if "chat_recs" not in st.session_state:
        st.session_state.chat_recs = {}  # turn index -> recommendations list
    if "chat_trace" not in st.session_state:
        st.session_state.chat_trace = {}  # turn index -> tool_trace list

    col_clear, col_mode = st.columns([1, 2])
    with col_clear:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_recs = {}
            st.session_state.chat_trace = {}
            st.rerun()
    with col_mode:
        agentic_mode = st.toggle(
            "🧠 Agentic mode (AI picks its own tools)",
            value=True,
            help=(
                "ON: the AI decides which tools to call (semantic search, "
                "year filter, statistics) based on your question.\n"
                "OFF: always uses RAG semantic search only."
            ),
        )

    # ---- Render existing conversation ----
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(
                f"<div class='poster-meta'>🧑 You</div>"
                f"<div class='ai-reply' style='border-left-color:#444'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='poster-meta'>🤖 StreamScope AI</div>"
                f"<div class='ai-reply'>{msg['content']}</div>",
                unsafe_allow_html=True
            )

            trace_info = st.session_state.chat_trace.get(i, {})
            tools_used = trace_info.get("tools", [])
            from_cache = trace_info.get("from_cache", False)

            caption_parts = []
            if from_cache:
                caption_parts.append("⚡ Cached response (instant, no API call)")
            if tools_used:
                tool_labels = " → ".join(t["tool"].replace("_tool", "") for t in tools_used)
                caption_parts.append(f"🔧 Tools used: {tool_labels}")

            if caption_parts:
                st.caption(" · ".join(caption_parts))

            recs = st.session_state.chat_recs.get(i, [])
            if recs:
                cols = st.columns(min(len(recs), 5))
                for j, r in enumerate(recs[:5]):
                    with cols[j]:
                        st.markdown(
                            f"""
                            <div class='poster-card'>
                                <span class='poster-badge'>{r.get('type', 'Title')}</span>
                                <div class='poster-title'>{r.get('title', 'Unknown')}</div>
                                <div class='poster-meta'>{r.get('year', '')} · {r.get('genre', '')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # ---- Input for the next turn ----
    q = st.text_input(
        "Ask AI",
        placeholder="I want a romantic movie in India",
        key=f"chat_input_{len(st.session_state.chat_history)}"
    )

    if st.button("Send", use_container_width=True) and q.strip():

        if agentic_mode:
            with st.spinner("StreamScope AI is deciding which tools to use..."):
                data = ask_ai_agentic(q, chat_history=st.session_state.chat_history)
        else:
            with st.spinner("StreamScope AI is searching the catalog..."):
                data = ask_ai_structured(q, chat_history=st.session_state.chat_history)

        reply_text = data.get("reply", "")
        recs = data.get("recommendations", [])
        trace = data.get("tool_trace", [])
        from_cache = data.get("from_cache", False)

        # Save this turn to history
        st.session_state.chat_history.append({"role": "user", "content": q})
        assistant_index = len(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
        st.session_state.chat_recs[assistant_index] = recs
        st.session_state.chat_trace[assistant_index] = {"tools": trace, "from_cache": from_cache}

        st.rerun()

# ======================================================
# MODEL LAB (XGBoost evaluation dashboard)
# ======================================================
elif "Model Lab" in page:

    st.subheader("📈 Model Lab — Movie vs TV Show Classifier")
    st.caption(
        "An XGBoost classifier trained on TF-IDF features (genre, director, "
        "cast, description) to predict whether a title is a Movie or a TV Show."
    )

    with st.spinner("Training model and computing metrics..."):
        results = train_xgboost()

    # ---- Metric cards ----
    m1, m2, m3, m4 = st.columns(4)
    metric_data = [
        ("Accuracy", results["accuracy"], "🎯"),
        ("Precision", results["precision"], "🔍"),
        ("Recall", results["recall"], "📡"),
        ("F1 Score", results["f1"], "⚖️"),
    ]
    for col, (label, value, icon) in zip([m1, m2, m3, m4], metric_data):
        col.markdown(
            f"""
            <div class='poster-card' style='text-align:center;'>
                <div style='font-size:26px;'>{icon}</div>
                <div style='font-size:28px; font-weight:800; color:#E50914;'>{value:.1%}</div>
                <div class='poster-meta'>{label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"<p class='poster-meta' style='margin-top:14px;'>"
        f"Trained on {results['train_size']} samples · Evaluated on {results['test_size']} held-out samples · "
        f"ROC AUC: <b style='color:#E50914;'>{results['auc']:.3f}</b></p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    col_cm, col_roc = st.columns(2)

    # ---- Confusion matrix heatmap ----
    with col_cm:
        st.markdown("#### Confusion Matrix")

        cm = results["cm"]
        labels = ["Movie", "TV Show"]

        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale=["#101010", "#E50914"],
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig_cm.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ---- ROC curve ----
    with col_roc:
        st.markdown("#### ROC Curve")

        fig_roc = px.area(
            x=results["fpr"],
            y=results["tpr"],
            labels=dict(x="False Positive Rate", y="True Positive Rate"),
        )
        fig_roc.add_shape(
            type="line", line=dict(dash="dash", color="#666"),
            x0=0, x1=1, y0=0, y1=1
        )
        fig_roc.update_traces(line_color="#E50914", fillcolor="rgba(229,9,20,0.15)")
        fig_roc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ---- Feature importance ----
    st.markdown("#### 🔑 Top Predictive Features")
    st.caption("Which words most influence the model's Movie vs TV Show prediction")

    fi_df = pd.DataFrame(results["feature_importance"])
    fig_fi = px.bar(
        fi_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#444", "#E50914"],
    )
    fig_fi.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

else:
    st.warning("Unknown page selected.")

# Floating Bot
st.markdown(
    "<div class='chatfab'>🤖</div>",
    unsafe_allow_html=True
)