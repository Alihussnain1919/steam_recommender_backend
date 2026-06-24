# ============================================================
# recommender.py — Content-based hybrid recommender
# ============================================================
# Input:  models/embeddings.npy, models/embeddings_ids.csv,
#         data/processed/steam_games_clean.csv
# Output: a function recommend(game_name) -> top N similar games
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


def load_everything(
    clean_path="data/processed/steam_games_clean.csv",
    embeddings_path="models/embeddings.npy",
    ids_path="models/embeddings_ids.csv"
):
    """
    Loads the full dataset and the embeddings, and merges them
    so each row has both its metadata AND its vector available.
    """

    df = pd.read_csv(clean_path)
    embeddings = np.load(embeddings_path)
    ids_df = pd.read_csv(ids_path)

    # Safety check: embeddings and ids_df must be in the same order
    # as df, otherwise everything downstream is silently wrong.
    assert len(df) == len(embeddings) == len(ids_df), \
        "Mismatch between dataset, embeddings, and id mapping — re-run Day 4."

    # Re-order df to guarantee it matches embeddings row-for-row
    # (in case the CSV got sorted differently somewhere)
    df = df.set_index("app_id").loc[ids_df["app_id"]].reset_index()

    print(f"Loaded {len(df)} games with {embeddings.shape[1]}-dim embeddings")

    return df, embeddings


def compute_numeric_similarity(df):
    """
    Builds a similarity matrix based on price and metacritic score
    closeness between every pair of games.

    Why: two games with similar price points and similar critical
    reception "feel" similar in a way pure text doesn't capture —
    e.g. two cheap, well-reviewed indie games.

    Returns: a (N, N) matrix where [i, j] = numeric similarity
             between game i and game j, scaled 0 to 1.
    """

    # Build a small feature matrix: [price, metacritic_normalized]
    # We use metacritic_normalized (0-1) from Day 2's fix instead
    # of raw score, so missing scores (0.5 neutral) don't distort things.
    price = df["price_eur"].fillna(0).values.reshape(-1, 1)
    meta  = df["metacritic_normalized"].fillna(0.5).values.reshape(-1, 1)

    # Normalize price to 0-1 range (min-max scaling)
    # Why: price ranges 0-80, metacritic is already 0-1.
    # Without scaling, price would dominate the similarity score
    # just because its numbers are bigger.
    price_scaled = (price - price.min()) / (price.max() - price.min() + 1e-9)
    # +1e-9 avoids division by zero if all prices were identical

    numeric_features = np.hstack([price_scaled, meta])
    # np.hstack stacks columns side by side: shape becomes (999, 2)

    # Cosine similarity on these 2D points
    numeric_sim = cosine_similarity(numeric_features)

    return numeric_sim


def build_similarity_matrices(df, embeddings):
    """
    Builds two similarity matrices:
      1. text_sim    — from the embeddings (genre/style/theme)
      2. numeric_sim — from price/metacritic
    """

    print("Computing text/tag embedding similarity...")
    text_sim = cosine_similarity(embeddings)
    # text_sim is (999, 999) — every game compared to every other game

    print("Computing numeric similarity (price, metacritic)...")
    numeric_sim = compute_numeric_similarity(df)

    return text_sim, numeric_sim


def hybrid_similarity(text_sim, numeric_sim, text_weight=0.8, numeric_weight=0.2):
    """
    Combines text similarity and numeric similarity into one score.

    text_weight, numeric_weight: how much each matrix contributes.
    These are your HYPERPARAMETERS — Day 7 will tune these with Optuna.
    Default 0.8/0.2 means text/genre/tags matter far more than price.
    """

    assert abs(text_weight + numeric_weight - 1.0) < 1e-6, \
        "Weights must sum to 1.0"

    combined = (text_weight * text_sim) + (numeric_weight * numeric_sim)
    return combined


def recommend(game_name, df, similarity_matrix, top_n=5):
    """
    Given a game name, returns the top_n most similar games.

    game_name        : exact or partial name of a game in the dataset
    df                : the games dataframe (must align with similarity_matrix rows)
    similarity_matrix : precomputed (N, N) similarity scores
    top_n             : how many recommendations to return
    """

    # Find the game — case-insensitive partial match, so user doesn't
    # need to type the exact capitalization
    matches = df[df["name"].str.lower().str.contains(game_name.lower(), na=False)]

    if len(matches) == 0:
        print(f"No game found matching '{game_name}'")
        return None

    if len(matches) > 1:
        print(f"Multiple matches found, using the first: '{matches.iloc[0]['name']}'")

    idx = matches.index[0]
    # idx is the row number of our target game in df — this MUST
    # correspond to the same row number in similarity_matrix

    scores = similarity_matrix[idx]
    # scores is a 1D array: similarity of our target game to all 999 games
    # (including itself, which will have similarity 1.0)

    # argsort sorts ascending; we want descending, so we reverse with [::-1]
    # We skip the first result because it's always the game itself
    similar_indices = scores.argsort()[::-1][1:top_n+1]

    results = df.iloc[similar_indices][
        ["name", "genres", "tags", "price_eur", "metacritic_score"]
    ].copy()
    results["similarity_score"] = scores[similar_indices].round(3)

    print(f"\nTop {top_n} games similar to '{matches.iloc[0]['name']}':\n")
    print(results.to_string(index=False))

    return results


# ============================================================
# Run a test
# ============================================================

if __name__ == "__main__":
    df, embeddings = load_everything()

    text_sim, numeric_sim = build_similarity_matrices(df, embeddings)

    final_sim = hybrid_similarity(text_sim, numeric_sim,
                                   text_weight=0.8, numeric_weight=0.2)

    # Save the similarity matrix so the frontend doesn't have to
    # recompute it every time (it's expensive for large datasets)
    os.makedirs("models", exist_ok=True)
    np.save("models/similarity_matrix.npy", final_sim)
    print("\nSaved similarity matrix to models/similarity_matrix.npy")

    # Test with a well-known game — change this to a game YOU
    # know is in your dataset
    recommend("Hades", df, final_sim, top_n=5)
    print("\n" + "="*60 + "\n")
    recommend("Counter-Strike", df, final_sim, top_n=5)