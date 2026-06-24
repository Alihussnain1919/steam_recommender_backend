# ============================================================
# api.py — FastAPI server exposing the recommender as a web API
# ============================================================
# This file does NOT reimplement your recommender logic.
# It just wraps your existing recommender.py functions so
# React (or anything else) can call them over HTTP.
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from src.recommender import load_everything,build_similarity_matrices, hybrid_similarity
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# Create the FastAPI app object
# ------------------------------------------------------------
# This `app` object is what uvicorn will run. Every endpoint
# you define below gets attached to this object.
app = FastAPI(title="SteamSense API")

# ------------------------------------------------------------
# CORS — Cross-Origin Resource Sharing
# ------------------------------------------------------------
# WHY THIS IS NEEDED: React runs on localhost:3000, FastAPI runs
# on localhost:8000. Browsers BLOCK requests between different
# ports by default for security (this is called the "same-origin
# policy"). CORSMiddleware tells the browser "it's fine, I trust
# requests from this other address." Without this, your React
# app's requests would fail silently with a CORS error in the
# browser console — a very common first-timer confusion.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173","https://semanticstream.netlify.app"],
    # 3000 = Create React App default, 5173 = Vite default
    
    # (we'll use Vite — explained below)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Load data ONCE when the server starts
# ------------------------------------------------------------
# WHY: load_everything() and build_similarity_matrices() are
# expensive (loading embeddings, computing a 999x999 matrix).
# If we ran this inside every endpoint, every single API call
# would take seconds. Instead we run it ONCE here, at import
# time, and keep the result in memory for the server's lifetime.
print("Loading data and building similarity matrix (this happens once)...")
df, embeddings = load_everything()
text_sim, numeric_sim = build_similarity_matrices(df, embeddings)
similarity_matrix = hybrid_similarity(text_sim, numeric_sim, text_weight=0.8, numeric_weight=0.2)
print("Loading embedding model for free-text search...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready.")



# ------------------------------------------------------------
# Endpoint 1: health check
# ------------------------------------------------------------
# WHY: a simple endpoint to confirm the server is alive.
# When you visit http://localhost:8000/ in your browser, you
# should see this JSON response.
@app.get("/")
def root():
    return {"status": "SteamSense API is running", "games_loaded": len(df)}


# ------------------------------------------------------------
# Endpoint 2: list all game names (for the React dropdown)
# ------------------------------------------------------------
@app.get("/games")
def list_games():
    """
    Returns every game name in the dataset.
    React calls this once when the page loads, to fill the dropdown.
    """
    names = sorted(df["name"].tolist())
    return {"games": names}

# ------------------------------------------------------------
# Endpoint 3: get genres
# ------------------------------------------------------------

@app.get("/genres")
def get_genres():
    genres = sorted(set(
        g.strip()
        for genres in df["genres"].dropna()
        for g in genres.split(",")
        if g.strip()
    ))

    return {"genres": genres}

# ------------------------------------------------------------
# Endpoint 3: get game detail on the base of id
# ------------------------------------------------------------

@app.get("/game/{app_id}")
def get_game(app_id: int):

    game = df[df["app_id"] == app_id]

    if len(game) == 0:
        raise HTTPException(404, "Game not found")

    row = game.iloc[0]

    row = game.iloc[0].replace({np.nan: None})

    return {
        "app_id": int(row["app_id"]),
        "name": row["name"],
        "short_description": row["short_description"],
        "detailed_description": row["detailed_description"],
        "genres": row["genres"],
        "price_eur": row["price_eur"],
        "metacritic_score": row["metacritic_score"],
        "release_date": row["release_date"],
        "developers": row["developers"],
        "publishers": row["publishers"],
        "header_image": row["header_image"]
    }


@app.get("/search")
def search_by_text(query: str, top_n: int = 5, max_price: float = 80.0, genre: str = "All"):
    """
    Lets the user type ANY free text (e.g. "relaxing farming game"
    or "scary survival horror with friends") instead of picking
    an exact game name.

    HOW: we embed the typed text with the SAME model used for
    games, then compare that single vector against every game's
    text embedding using cosine similarity — no retraining needed,
    because the model already maps similar meaning to nearby vectors
    regardless of whether the text came from a game description or
    a user query.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Encode the user's text into the same 384-dim space as the games
    query_vector = text_model.encode([query],convert_to_numpy=True)
    # shape: (1, 384) — one vector, since we passed one string

    # Compare against the TEXT-ONLY embeddings (the `embeddings`
    # array from Day 4), not the hybrid matrix — the hybrid matrix
    # is game-vs-game, but here we have a NEW vector that isn't
    # one of the 999 games, so we compute similarity fresh.
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(query_vector, embeddings)[0]
    # sims shape: (999,) — similarity of the query to every game

    ranked_indices = sims.argsort()[::-1]

    results = df.iloc[ranked_indices].copy()
    results["similarity_score"] = sims[ranked_indices]

    if genre != "All":
        results = results[results["genres"].str.contains(genre, na=False)]
    results = results[results["price_eur"] <= max_price]
    results = results.head(top_n)

    results = results.replace({np.nan: None})
    output = results[
        [
        "app_id",
        "name",
        "genres",
        "tags",
        "price_eur",
        "metacritic_score",
        "similarity_score",
        "header_image"
        ]
    ].to_dict("records")

    return {"query_game":  query, "recommendations": output}


# ------------------------------------------------------------
# Endpoint 4: get recommendations for a game
# ------------------------------------------------------------
@app.get("/recommend")
def get_recommendations(game: str, top_n: int = 5, max_price: float = 80.0, genre: str = "All"):
    """
    Query parameters (these come from the URL, e.g.
    /recommend?game=Hades&top_n=5&max_price=30):

    game      : exact name of the game the user picked
    top_n     : how many recommendations to return
    max_price : filter out recommendations above this price
    genre     : "All" or a specific genre to filter by
    """

    matches = df[df["name"] == game]

    if len(matches) == 0:
        # HTTPException sends a proper error response (404 = not found)
        # instead of crashing the server
        raise HTTPException(status_code=404, detail=f"Game '{game}' not found")

    idx = matches.index[0]
    scores = similarity_matrix[idx]
    ranked_indices = scores.argsort()[::-1]
    ranked_indices = [r for r in ranked_indices if r != idx]

    results = df.iloc[ranked_indices].copy()
    results["similarity_score"] = scores[ranked_indices]

    # Apply filters
    if genre != "All":
        results = results[results["genres"].str.contains(genre, na=False)]
    results = results[results["price_eur"] <= max_price]

    results = results.head(top_n)

    # Convert to a list of dictionaries — this is what becomes JSON
    # automatically when FastAPI sends the response. You don't need
    # to manually convert to JSON; FastAPI does it for you.
    results = results.replace({np.nan: None})
    output = results[
        [
        "app_id",
        "name",
        "genres",
        "tags",
        "price_eur",
        "metacritic_score",
        "similarity_score",
        "header_image"
        ]
    ].to_dict("records")

    return {"query_game": game, "recommendations": output}