import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix AI Dashboard", layout="wide")


st.markdown("""
<style>
body {background:#0b0b0b; color:white;}
h1,h2,h3 {color:#E50914;}
.card {
    background:#141414;
    padding:20px;
    border-radius:12px;
    text-align:center;
    box-shadow:0 4px 20px rgba(229,9,20,0.3);
}
section[data-testid="stSidebar"] {
    background:#111;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    url = 'https://drive.google.com/uc?export=download&id=1uCSB6lS329wnOLrQCKHOj3ZxWYM6MGX-'
    df = pd.read_csv(io.StringIO(requests.get(url).content.decode('utf-8')))
    df['country'] = df['country'].fillna('Unknown')
    df['listed_in'] = df['listed_in'].fillna('')
    df['primary_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0] if x else 'Unknown')
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    return df

df = load_data()

page = st.sidebar.radio("📌 Navigation", ["Dashboard", "Recommendation"])

if page == "Dashboard":

    st.title("🎬 Netflix Interactive Dashboard")

    # -------- FILTERS --------
    st.sidebar.header(" Filters")

    type_filter = st.sidebar.multiselect("Type", df['type'].unique(), default=df['type'].unique())

    year_range = st.sidebar.slider(
        "Release Year",
        int(df['release_year'].min()),
        int(df['release_year'].max()),
        (2000, 2020)
    )

    country_filter = st.sidebar.multiselect(
        "Country",
        df['country'].value_counts().head(15).index.tolist()
    )

    genre_filter = st.sidebar.multiselect(
        "Genre",
        df['primary_genre'].unique()
    )

    # -------- APPLY FILTER --------
    filtered_df = df[
        (df['type'].isin(type_filter)) &
        (df['release_year'] >= year_range[0]) &
        (df['release_year'] <= year_range[1])
    ]

    if country_filter:
        filtered_df = filtered_df[filtered_df['country'].str.contains('|'.join(country_filter))]

    if genre_filter:
        filtered_df = filtered_df[filtered_df['primary_genre'].isin(genre_filter)]

    # -------- KPI --------
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card'><h3>Total</h3><h2>{len(filtered_df)}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h3>Movies</h3><h2>{len(filtered_df[filtered_df['type']=='Movie'])}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h3>TV Shows</h3><h2>{len(filtered_df[filtered_df['type']=='TV Show'])}</h2></div>", unsafe_allow_html=True)

    st.divider()

    # -------- CHARTS --------
    colA, colB = st.columns(2)

    with colA:
        fig1 = px.pie(filtered_df, names='type',
                      title="Content Type",
                      color_discrete_sequence=['#E50914', '#333'])
        st.plotly_chart(fig1, width='stretch')

    with colB:
        genre = filtered_df['primary_genre'].value_counts().head(8)
        fig2 = px.bar(genre, x=genre.values, y=genre.index,
                      orientation='h',
                      title="Top Genres",
                      color=genre.values,
                      color_continuous_scale='Reds')
        st.plotly_chart(fig2, width='stretch')

    colC, colD = st.columns(2)

    with colC:
        country = filtered_df['country'].value_counts().head(10)
        fig3 = px.bar(country, x=country.values, y=country.index,
                      orientation='h',
                      title="Top Countries",
                      color=country.values,
                      color_continuous_scale='Reds')
        st.plotly_chart(fig3, width='stretch')

    with colD:
        growth = filtered_df.groupby('year_added').size().reset_index(name='count')
        fig4 = px.line(growth, x='year_added', y='count',
                       title="Growth Over Time",
                       color_discrete_sequence=['#E50914'])
        st.plotly_chart(fig4, width='stretch')

else:

    st.title("🌟 Smart Recommendation System")

    df = df.dropna(subset=['title'])

    # -------- TF-IDF FEATURES --------
    df['features'] = df['listed_in'] + " " + df['type']

    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['features'])

    # -------- SELECT MOVIE --------
    movie_list = df['title'].unique()
    selected_movie = st.selectbox("🎬 Select a Movie", movie_list)

    def recommend(movie):
        idx = df[df['title'] == movie].index[0]

        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

        top_indices = sim_scores.argsort()[-6:-1][::-1]

        return df.iloc[top_indices]['title'].values

    if st.button("🔥 Recommend"):
        recs = recommend(selected_movie)

        st.subheader("🎬 Recommended for You")

        cols = st.columns(5)
        for i, movie in enumerate(recs):
            with cols[i]:
                st.markdown(f"<div class='card'><h4>{movie}</h4></div>", unsafe_allow_html=True)