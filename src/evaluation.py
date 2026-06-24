# ============================================================
# evaluation.py — Precision@k / Recall@k with two ground truths
# ============================================================

import pandas as pd
import numpy as np


def build_tag_ground_truth(df, min_shared_tags=3):
    """
    Method 1: Game B is "relevant" to game A if they share
    at least `min_shared_tags` tags.

    Returns: dict {game_index: set(relevant_game_indices)}
    """
    tag_sets = []
    for tags_str in df["tags"]:
        tags = set(t.strip() for t in str(tags_str).split(",") if t.strip())
        tag_sets.append(tags)

    ground_truth = {}
    n = len(df)
    for i in range(n):
        relevant = set()
        for j in range(n):
            if i == j:
                continue
            shared = len(tag_sets[i] & tag_sets[j])
            if shared >= min_shared_tags:
                relevant.add(j)
        ground_truth[i] = relevant

    return ground_truth


def build_genre_price_ground_truth(df, price_tolerance=15.0):
    """
    Method 2: Game B is "relevant" to game A if they share at
    least one genre AND their price difference is within
    `price_tolerance` euros.

    This is intentionally a DIFFERENT rule from Method 1, so the
    two evaluations are not just measuring the same thing twice.
    """
    genre_sets = []
    for genre_str in df["genres"]:
        genres = set(g.strip() for g in str(genre_str).split(",") if g.strip())
        genre_sets.append(genres)

    prices = df["price_eur"].fillna(0).values

    ground_truth = {}
    n = len(df)
    for i in range(n):
        relevant = set()
        for j in range(n):
            if i == j:
                continue
            shares_genre = len(genre_sets[i] & genre_sets[j]) > 0
            close_price = abs(prices[i] - prices[j]) <= price_tolerance
            if shares_genre and close_price:
                relevant.add(j)
        ground_truth[i] = relevant

    return ground_truth


def precision_at_k(recommended_indices, relevant_set, k):
    """
    Of the top-k recommended games, what fraction are relevant?
    """
    top_k = recommended_indices[:k]
    if len(top_k) == 0:
        return 0.0
    hits = len(set(top_k) & relevant_set)
    return hits / k


def recall_at_k(recommended_indices, relevant_set, k):
    """
    Of all the relevant games, what fraction did we find in top-k?
    """
    if len(relevant_set) == 0:
        return None  # undefined — game has no relevant games at all
    top_k = recommended_indices[:k]
    hits = len(set(top_k) & relevant_set)
    return hits / len(relevant_set)


def evaluate_recommender(similarity_matrix, ground_truth, df, k=5, sample_size=200):
    """
    Runs precision/recall across a SAMPLE of games (not all 999 —
    that would be 999x999 comparisons, slow for no extra insight).
    Returns average precision@k and recall@k.
    """
    n = len(df)
    sample_indices = np.random.choice(n, size=min(sample_size, n), replace=False)

    precisions, recalls = [], []

    for idx in sample_indices:
        scores = similarity_matrix[idx]
        recommended = scores.argsort()[::-1]
        recommended = [r for r in recommended if r != idx][:k]

        relevant = ground_truth[idx]

        p = precision_at_k(recommended, relevant, k)
        r = recall_at_k(recommended, relevant, k)

        precisions.append(p)
        if r is not None:
            recalls.append(r)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls) if recalls else 0.0

    return avg_precision, avg_recall


if __name__ == "__main__":
    df = pd.read_csv("data/processed/steam_games_clean.csv")
    similarity_matrix = np.load("models/similarity_matrix.npy")

    print("Building ground truth (Method 1: tag overlap)...")
    gt_tags = build_tag_ground_truth(df, min_shared_tags=3)

    print("Building ground truth (Method 2: genre + price)...")
    gt_genre_price = build_genre_price_ground_truth(df, price_tolerance=15.0)

    print("\nEvaluating with tag-overlap ground truth...")
    p1, r1 = evaluate_recommender(similarity_matrix, gt_tags, df, k=5)
    print(f"  Precision@5: {p1:.3f}   Recall@5: {r1:.3f}")

    print("\nEvaluating with genre+price ground truth...")
    p2, r2 = evaluate_recommender(similarity_matrix, gt_genre_price, df, k=5)
    print(f"  Precision@5: {p2:.3f}   Recall@5: {r2:.3f}")