# ============================================================
# streamlit_app.py — SteamSense frontend
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from recommender import load_everything, build_similarity_matrices, hybrid_similarity

st.set_page_config(page_title="SteamSense", page_icon="🎮", layout="wide")


@st.cache_data
def load_data():
    df, embeddings = load_everything()
    text_sim, numeric_sim = build_similarity_matrices(df, embeddings)
    final_sim = hybrid_similarity(text_sim, numeric_sim, text_weight=0.8, numeric_weight=0.2)
    return df, final_sim
    # @st.cache_data means this expensive function only runs ONCE,
    # even if the user interacts with the app many times. Without
    # this, Streamlit re-runs the whole script on every click.


st.title("🎮 SteamSense")
st.caption("A hybrid content-based game recommender built on Steam data")

df, similarity_matrix = load_data()

# ---- Sidebar filters ----
st.sidebar.header("Filters")
selected_genre = st.sidebar.selectbox(
    "Filter by genre (optional)",
    options=["All"] + sorted(set(g.strip() for genres in df["genres"].dropna()
                                   for g in genres.split(",") if g.strip()))
)
max_price = st.sidebar.slider("Maximum price (€)", 0, 80, 80)
top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

# ---- Game selector ----
game_name = st.selectbox("Choose a game you like:", sorted(df["name"].tolist()))

if st.button("Get recommendations", type="primary"):
    idx = df[df["name"] == game_name].index[0]
    scores = similarity_matrix[idx]
    ranked_indices = scores.argsort()[::-1]
    ranked_indices = [r for r in ranked_indices if r != idx]

    results = df.iloc[ranked_indices].copy()
    results["similarity_score"] = scores[ranked_indices]

    # Apply sidebar filters
    if selected_genre != "All":
        results = results[results["genres"].str.contains(selected_genre, na=False)]
    results = results[results["price_eur"] <= max_price]

    results = results.head(top_n)

    st.subheader(f"Games similar to {game_name}")

    for _, row in results.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            if row.get("header_image"):
                st.image(row["header_image"], use_container_width=True)
        with col2:
            st.markdown(f"**{row['name']}**  —  €{row['price_eur']:.2f}")
            st.caption(f"Genres: {row['genres']}")
            st.caption(f"Tags: {row['tags']}")
            st.progress(min(1.0, max(0.0, row["similarity_score"])))
            st.caption(f"Similarity score: {row['similarity_score']:.3f}")
        st.divider()






# # ============================================================
# # streamlit_app.py — SteamSense (Premium UI Version)
# # ============================================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# from recommender import load_everything, build_similarity_matrices, hybrid_similarity

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(
#     page_title="SteamSense AI",
#     page_icon="🎮",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ---------------- GLOBAL STYLE ----------------
# st.markdown("""
# <style>

# /* Background */
# .stApp {
#     background: radial-gradient(circle at top, #0b1220, #05070d);
#     color: #ffffff;
# }

# /* Title */
# .main-title {
#     font-size: 44px;
#     font-weight: 800;
#     background: linear-gradient(90deg, #6ee7ff, #a78bfa);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
# }

# /* Subtitle */
# .subtitle {
#     font-size: 15px;
#     color: #9ca3af;
#     margin-bottom: 25px;
# }

# /* Card */
# .game-card {
#     background: rgba(255,255,255,0.04);
#     border: 1px solid rgba(255,255,255,0.08);
#     border-radius: 16px;
#     padding: 15px;
#     backdrop-filter: blur(10px);
# }

# /* Button */
# .stButton > button {
#     background: linear-gradient(90deg, #6ee7ff, #a78bfa);
#     color: black;
#     font-weight: 700;
#     border-radius: 10px;
#     padding: 10px 20px;
#     border: none;
# }

# .stButton > button:hover {
#     transform: scale(1.02);
#     transition: 0.2s;
# }

# </style>
# """, unsafe_allow_html=True)


# # ---------------- CACHE ----------------
# @st.cache_data
# def load_data():
#     df, embeddings = load_everything()
#     text_sim, numeric_sim = build_similarity_matrices(df, embeddings)
#     final_sim = hybrid_similarity(text_sim, numeric_sim, 0.8, 0.2)
#     return df, final_sim


# # ---------------- HEADER ----------------
# st.markdown('<div class="main-title">🎮 SteamSense AI</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle">Next-gen AI Game Recommendation Engine</div>', unsafe_allow_html=True)

# df, similarity_matrix = load_data()


# # ---------------- SIDEBAR ----------------
# st.sidebar.header("⚙️ Control Panel")

# selected_genre = st.sidebar.selectbox(
#     "Genre Filter",
#     ["All"] + sorted(set(
#         g.strip()
#         for genres in df["genres"].dropna()
#         for g in genres.split(",")
#         if g.strip()
#     ))
# )

# max_price = st.sidebar.slider("Max Price (€)", 0, 80, 80)
# top_n = st.sidebar.slider("Results", 3, 12, 5)

# st.sidebar.markdown("---")
# st.sidebar.info("💡 Tip: Try indie games like Hades, Celeste, Hollow Knight")


# # ---------------- INPUT ----------------
# game_name = st.selectbox("🎮 Choose your game", sorted(df["name"].tolist()))

# run = st.button("✨ Generate AI Recommendations")


# # ---------------- MAIN LOGIC ----------------
# if run:

#     idx = df[df["name"] == game_name].index[0]
#     scores = similarity_matrix[idx]

#     ranked_indices = scores.argsort()[::-1]
#     ranked_indices = [i for i in ranked_indices if i != idx]

#     results = df.iloc[ranked_indices].copy()
#     results["score"] = scores[ranked_indices]

#     if selected_genre != "All":
#         results = results[results["genres"].str.contains(selected_genre, na=False)]

#     results = results[results["price_eur"] <= max_price]
#     results = results.head(top_n)

#     # ---------------- HERO SECTION ----------------
#     st.markdown(f"## 🔥 Because you liked **{game_name}**")

#     # ---------------- RESULTS GRID ----------------
#     for _, row in results.iterrows():

#         with st.container():

#             col1, col2, col3 = st.columns([1, 5, 1])

#             with col1:
#                 if row.get("header_image"):
#                     st.image(row["header_image"], use_container_width=True)

#             with col2:
#                 st.markdown(f"### 🎮 {row['name']}")

#                 st.markdown(f"""
#                 <div style="color:#9ca3af;font-size:13px;">
#                 🎭 {row['genres']} <br>
#                 🏷️ {row['tags'][:120]}...
#                 </div>
#                 """, unsafe_allow_html=True)

#                 st.progress(float(min(max(row["score"], 0), 1)))

#                 st.caption(f"Similarity Score: {row['score']:.3f}")

#             with col3:
#                 st.markdown(f"### 💰 €{row['price_eur']:.2f}")

#                 if row['price_eur'] == 0:
#                     st.success("FREE")
#                 elif row['price_eur'] < 10:
#                     st.info("LOW")
#                 else:
#                     st.warning("PAID")

#         st.markdown("---")        