import streamlit as st, pandas as pd, requests, io, plotly.express as px
from model import train_xgboost, get_recommendations
from ai_helper import ask_ai

st.set_page_config(page_title='Netflix AI', layout='wide', page_icon='🎬')

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
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    url = "https://drive.google.com/uc?export=download&id=1uCSB6lS329wnOLrQCKHOj3ZxWYM6MGX-"

    df = pd.read_csv(
        io.StringIO(
            requests.get(url).content.decode("utf-8")
        )
    )

    df["country"] = df["country"].fillna("Unknown")
    df["listed_in"] = df["listed_in"].fillna("Unknown")
    df["year_added"] = pd.to_datetime(
        df["date_added"],
        errors="coerce"
    ).dt.year

    return df


df = load()

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

nav = st.columns(4)

for c, p in zip(
    nav,
    ["🏠 Dashboard", "🎬 Discover", "📊 Analytics", "🤖 AI Lab"]
):
    if c.button(p, use_container_width=True, key=p):
        st.session_state.page = p

page = st.session_state.page

st.markdown(
    "# <span style='color:#E50914'>NETFLIX</span> 🤖 AI",
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

        cols = st.columns(5)

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
# AI LAB
# ======================================================
else:

    st.subheader("🤖 AI Assistant")

    q = st.text_input(
        "Ask AI",
        placeholder="I want romantic movie in India"
    )

    if st.button("Send", use_container_width=True):

        ans = ask_ai(q)

        st.markdown(
            f"<div class='card'>{ans}</div>",
            unsafe_allow_html=True
        )

# Floating Bot
st.markdown(
    "<div class='chatfab'>🤖</div>",
    unsafe_allow_html=True
)