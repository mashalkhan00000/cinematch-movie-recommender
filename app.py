import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e6e1;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #1a0a2e 0%, #0a0a0f 50%),
                radial-gradient(ellipse at 80% 80%, #0d1a2e 0%, transparent 60%);
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-tag {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #c084fc;
    background: rgba(192,132,252,0.1);
    border: 1px solid rgba(192,132,252,0.25);
    padding: 0.3rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 900;
    line-height: 1.05;
    background: linear-gradient(135deg, #fff 30%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.8rem;
}
.hero-sub {
    color: #9ca3af;
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
}

/* Search area */
.search-wrap {
    max-width: 600px;
    margin: 0 auto 1rem;
}

/* Streamlit input override */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #e8e6e1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.2rem 0.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(192,132,252,0.5) !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(168,85,247,0.4) !important;
}

/* Movie cards */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    margin: 2.5rem 0 1.2rem;
    padding-left: 0.2rem;
    border-left: 3px solid #a855f7;
    padding-left: 0.8rem;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1.2rem;
    margin-bottom: 2rem;
}

.movie-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #c084fc);
    opacity: 0;
    transition: opacity 0.25s;
}
.movie-card:hover {
    border-color: rgba(168,85,247,0.35);
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    background: rgba(255,255,255,0.07);
}
.movie-card:hover::before { opacity: 1; }

.card-rank {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a855f7;
    margin-bottom: 0.6rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f3f0ea;
    margin-bottom: 0.5rem;
    line-height: 1.3;
}
.card-year {
    font-size: 0.78rem;
    color: #6b7280;
    margin-bottom: 0.7rem;
}
.genre-pill {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    color: #c084fc;
    background: rgba(192,132,252,0.1);
    border: 1px solid rgba(192,132,252,0.2);
    padding: 0.2rem 0.55rem;
    border-radius: 100px;
    margin: 0.15rem 0.1rem 0 0;
}
.score-bar-wrap {
    margin-top: 1rem;
}
.score-label {
    font-size: 0.7rem;
    color: #6b7280;
    margin-bottom: 0.3rem;
    display: flex;
    justify-content: space-between;
}
.score-bar {
    height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #7c3aed, #c084fc);
}

/* Selected movie highlight */
.selected-movie {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(168,85,247,0.08));
    border: 1px solid rgba(168,85,247,0.3);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.selected-icon { font-size: 2.5rem; }
.selected-info h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #fff;
    margin-bottom: 0.3rem;
}
.selected-info p { font-size: 0.85rem; color: #9ca3af; }

/* Stats row */
.stat-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.stat-box {
    flex: 1;
    min-width: 120px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #a855f7;
}
.stat-label { font-size: 0.72rem; color: #6b7280; margin-top: 0.2rem; }

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 2rem 0;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 1rem;
    color: #374151;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ─── Dataset ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    movies = [
        {"title": "The Dark Knight", "year": 2008, "genres": "Action Crime Drama", "description": "Batman faces the Joker, a criminal mastermind who plunges Gotham into chaos and anarchy."},
        {"title": "Inception", "year": 2010, "genres": "Action Sci-Fi Thriller", "description": "A thief enters dreams to steal secrets from subconscious minds using dream-sharing technology."},
        {"title": "Interstellar", "year": 2014, "genres": "Adventure Drama Sci-Fi", "description": "Astronauts travel through a wormhole near Saturn to find a new home for humanity as Earth dies."},
        {"title": "The Matrix", "year": 1999, "genres": "Action Sci-Fi", "description": "A hacker discovers reality is a simulation and joins a rebellion against machine controllers."},
        {"title": "Pulp Fiction", "year": 1994, "genres": "Crime Drama Thriller", "description": "Intersecting stories of criminals, hitmen, and a boxer in Los Angeles told out of chronological order."},
        {"title": "The Shawshank Redemption", "year": 1994, "genres": "Drama", "description": "A banker is wrongly convicted and forms a friendship with a fellow prisoner while dreaming of freedom."},
        {"title": "Forrest Gump", "year": 1994, "genres": "Drama Romance", "description": "A simple man with low IQ witnesses and influences major historical events across several decades."},
        {"title": "The Silence of the Lambs", "year": 1991, "genres": "Crime Drama Thriller Horror", "description": "An FBI trainee seeks help from a brilliant imprisoned cannibal to catch a serial killer."},
        {"title": "Fight Club", "year": 1999, "genres": "Drama Thriller", "description": "An insomniac office worker forms an underground fight club with a soap salesman that spirals into chaos."},
        {"title": "Goodfellas", "year": 1990, "genres": "Crime Drama Biography", "description": "The rise and fall of a mob associate over several decades in the New York crime world."},
        {"title": "The Godfather", "year": 1972, "genres": "Crime Drama", "description": "An aging patriarch transfers control of his mafia empire to his reluctant youngest son."},
        {"title": "Avengers Endgame", "year": 2019, "genres": "Action Adventure Sci-Fi", "description": "The Avengers reassemble to reverse Thanos's destruction and restore the universe."},
        {"title": "Parasite", "year": 2019, "genres": "Drama Thriller Comedy", "description": "A poor family schemes to infiltrate a wealthy household with unexpected and violent consequences."},
        {"title": "Dune", "year": 2021, "genres": "Adventure Drama Sci-Fi", "description": "A noble family's heir arrives on a desert planet critical for a precious resource that controls the universe."},
        {"title": "Oppenheimer", "year": 2023, "genres": "Biography Drama History", "description": "The story of the theoretical physicist who led development of the first nuclear weapons during WWII."},
        {"title": "Whiplash", "year": 2014, "genres": "Drama Music", "description": "A young drummer pursues greatness under the tutelage of a brutal and demanding music conductor."},
        {"title": "La La Land", "year": 2016, "genres": "Drama Music Romance", "description": "A jazz musician and aspiring actress fall in love in Los Angeles while chasing their dreams."},
        {"title": "Get Out", "year": 2017, "genres": "Horror Mystery Thriller", "description": "A Black man uncovers a disturbing secret when he visits his white girlfriend's family estate."},
        {"title": "Mad Max Fury Road", "year": 2015, "genres": "Action Adventure Sci-Fi", "description": "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler with the help of a drifter."},
        {"title": "The Grand Budapest Hotel", "year": 2014, "genres": "Adventure Comedy Drama", "description": "A legendary concierge and his lobby boy become embroiled in theft, murder, and war in 1930s Europe."},
        {"title": "Blade Runner 2049", "year": 2017, "genres": "Action Drama Sci-Fi Mystery", "description": "A young blade runner discovers a secret that could plunge society into chaos as he searches for answers."},
        {"title": "1917", "year": 2019, "genres": "Action Drama War History", "description": "Two British soldiers race against time through enemy territory to deliver a message that could save 1600 lives."},
        {"title": "Joker", "year": 2019, "genres": "Crime Drama Thriller", "description": "A failed comedian descends into madness and becomes the iconic Gotham City villain."},
        {"title": "Spider-Man No Way Home", "year": 2021, "genres": "Action Adventure Sci-Fi", "description": "Peter Parker asks Doctor Strange for help after his identity is revealed, opening the multiverse."},
        {"title": "Tenet", "year": 2020, "genres": "Action Sci-Fi Thriller", "description": "A secret agent uses time inversion to prevent World War III in a story that moves both forward and backward."},
        {"title": "The Revenant", "year": 2015, "genres": "Action Adventure Drama", "description": "A frontiersman on a fur trading expedition fights for survival after being mauled by a bear."},
        {"title": "Gone Girl", "year": 2014, "genres": "Drama Mystery Thriller", "description": "A man becomes the prime suspect when his wife mysteriously disappears on their wedding anniversary."},
        {"title": "Shutter Island", "year": 2010, "genres": "Mystery Thriller Drama", "description": "A US Marshal investigates a psychiatric facility where a patient has gone missing."},
        {"title": "Prisoners", "year": 2013, "genres": "Crime Drama Mystery Thriller", "description": "A father takes drastic action when his daughter disappears and police can't find the suspect."},
        {"title": "No Country for Old Men", "year": 2007, "genres": "Crime Drama Thriller Western", "description": "A hunter stumbles upon drug money and is relentlessly pursued by a merciless hitman."},
    ]
    return pd.DataFrame(movies)

@st.cache_data
def build_model(df):
    df["combined"] = df["genres"] + " " + df["description"]
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    matrix = tfidf.fit_transform(df["combined"])
    similarity = cosine_similarity(matrix)
    return similarity

def get_recommendations(title, df, similarity, n=8):
    idx = df[df["title"] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    results = []
    for i, score in scores:
        results.append({**df.iloc[i].to_dict(), "score": round(score * 100, 1)})
    return results

# ─── Load ──────────────────────────────────────────────────────
df = load_data()
similarity = build_model(df)

# ─── UI ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ AI-Powered · Content-Based Filtering</div>
    <div class="hero-title">CineMatch</div>
    <div class="hero-sub">Discover your next obsession. Powered by TF-IDF & Cosine Similarity.</div>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown(f"""
<div class="stat-row">
    <div class="stat-box"><div class="stat-num">{len(df)}</div><div class="stat-label">Movies in Database</div></div>
    <div class="stat-box"><div class="stat-num">TF-IDF</div><div class="stat-label">Algorithm Used</div></div>
    <div class="stat-box"><div class="stat-num">8</div><div class="stat-label">Recommendations</div></div>
    <div class="stat-box"><div class="stat-num">∞</div><div class="stat-label">Discoveries Awaiting</div></div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# Search
col1, col2 = st.columns([4, 1])
with col1:
    selected = st.selectbox("", df["title"].sort_values().tolist(), index=0, label_visibility="collapsed", placeholder="Search a movie you love...")
with col2:
    recommend_btn = st.button("✦ Find Matches")

if recommend_btn or selected:
    movie_info = df[df["title"] == selected].iloc[0]
    genres_html = "".join([f'<span class="genre-pill">{g.strip()}</span>' for g in movie_info["genres"].split()])
    st.markdown(f"""
    <div class="selected-movie">
        <div class="selected-icon">🎬</div>
        <div class="selected-info">
            <h3>{movie_info['title']} <span style="font-size:0.8rem;color:#6b7280;font-family:'DM Sans',sans-serif;font-weight:400">({movie_info['year']})</span></h3>
            <p>{movie_info['description']}</p>
            <div style="margin-top:0.6rem">{genres_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    recs = get_recommendations(selected, df, similarity, n=8)

    st.markdown('<div class="section-title">Recommended For You</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-grid">', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, rec in enumerate(recs):
        genres_html = "".join([f'<span class="genre-pill">{g.strip()}</span>' for g in rec["genres"].split()])
        with cols[i % 4]:
            st.markdown(f"""
            <div class="movie-card">
                <div class="card-rank">Match #{i+1}</div>
                <div class="card-title">{rec['title']}</div>
                <div class="card-year">{rec['year']}</div>
                <div>{genres_html}</div>
                <div class="score-bar-wrap">
                    <div class="score-label"><span>Similarity Score</span><span style="color:#a855f7">{rec['score']}%</span></div>
                    <div class="score-bar"><div class="score-fill" style="width:{rec['score']}%"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Built with Python · Scikit-learn · Streamlit &nbsp;·&nbsp; Content-Based Filtering using TF-IDF + Cosine Similarity</div>', unsafe_allow_html=True)
