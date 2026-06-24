# ============================================================
# perturbation.py — Robustness testing via input corruption
# ============================================================
# We corrupt the dataset in 3 different ways, regenerate
# embeddings on the corrupted version, and compare the
# recommendations against the original (clean) recommendations.
# ============================================================

import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from recommender import load_everything, build_similarity_matrices, hybrid_similarity


def jaccard_similarity(set_a, set_b):
    """
    Measures overlap between two sets.
    1.0 = identical sets, 0.0 = no overlap at all.
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = set_a | set_b
    intersection = set_a & set_b
    return len(intersection) / len(union)


def get_top_k_indices(similarity_matrix, idx, k=5):
    """Returns the top-k most similar game indices for game `idx`."""
    scores = similarity_matrix[idx]
    ranked = scores.argsort()[::-1]
    ranked = [r for r in ranked if r != idx]
    return set(ranked[:k])


# ------------------------------------------------------------
# Perturbation 1: Remove tags
# ------------------------------------------------------------
def perturb_remove_tags(df, fraction=0.5):
    """
    Removes `fraction` of tags from every game.
    Tests: does the recommender rely too heavily on tags?
    """
    df_p = df.copy()

    def drop_some_tags(tags_str):
        tags = [t.strip() for t in str(tags_str).split(",") if t.strip()]
        keep_n = max(0, int(len(tags) * (1 - fraction)))
        kept = random.sample(tags, keep_n) if keep_n > 0 else []
        return ", ".join(kept)

    df_p["tags"] = df_p["tags"].apply(drop_some_tags)
    return df_p


# ------------------------------------------------------------
# Perturbation 2: Truncate descriptions
# ------------------------------------------------------------
def perturb_truncate_description(df, max_chars=40):
    """
    Cuts descriptions down to the first `max_chars` characters.
    Tests: does the recommender need full descriptions, or do
    a few words of signal carry most of the meaning?
    """
    df_p = df.copy()
    df_p["short_description"] = df_p["short_description"].str[:max_chars]
    return df_p


# ------------------------------------------------------------
# Perturbation 3: Shuffle genres
# ------------------------------------------------------------
def perturb_shuffle_genres(df, fraction=0.3):
    """
    Randomly reassigns genres for `fraction` of games to a
    RANDOM OTHER game's genres. Tests: how much does wrong/noisy
    genre data break the system?
    """
    df_p = df.copy()
    n = len(df_p)
    n_to_shuffle = int(n * fraction)
    shuffle_indices = random.sample(range(n), n_to_shuffle)

    genres_pool = df_p["genres"].tolist()
    for idx in shuffle_indices:
        df_p.loc[idx, "genres"] = random.choice(genres_pool)

    return df_p


# ------------------------------------------------------------
# Rebuild combined_text + embeddings for a perturbed dataset
# ------------------------------------------------------------
def rebuild_embeddings(df_p, model):
    def build_combined_text(row):
        parts = []
        if row["short_description"]:
            parts.append(row["short_description"])
        if row["genres"]:
            parts.append(f"Genre: {row['genres']}")
        if row["tags"]:
            parts.append(f"Tags: {row['tags']}")
        if row["categories"]:
            parts.append(f"Categories: {row['categories']}")
        return " | ".join(parts)

    df_p["combined_text"] = df_p.apply(build_combined_text, axis=1)
    embeddings_p = model.encode(df_p["combined_text"].tolist(),
                                 show_progress_bar=False, batch_size=32)
    return embeddings_p, df_p


# ------------------------------------------------------------
# Main robustness test
# ------------------------------------------------------------
def run_perturbation_analysis(sample_size=50, k=5):
    print("Loading original data and similarity matrix...")
    df, embeddings = load_everything()
    text_sim, numeric_sim = build_similarity_matrices(df, embeddings)
    original_sim = hybrid_similarity(text_sim, numeric_sim, 0.8, 0.2)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    perturbations = {
        "Remove 50% of tags": perturb_remove_tags(df, fraction=0.5),
        "Truncate descriptions to 40 chars": perturb_truncate_description(df, max_chars=40),
        "Shuffle 30% of genres": perturb_shuffle_genres(df, fraction=0.3),
    }

    n = len(df)
    sample_indices = np.random.choice(n, size=min(sample_size, n), replace=False)

    results = {}

    for name, df_perturbed in perturbations.items():
        print(f"\nTesting: {name}")
        emb_p, df_p = rebuild_embeddings(df_perturbed, model)
        text_sim_p, numeric_sim_p = build_similarity_matrices(df_p, emb_p)
        sim_p = hybrid_similarity(text_sim_p, numeric_sim_p, 0.8, 0.2)

        jaccard_scores = []
        for idx in sample_indices:
            before = get_top_k_indices(original_sim, idx, k=k)
            after = get_top_k_indices(sim_p, idx, k=k)
            jaccard_scores.append(jaccard_similarity(before, after))

        avg_jaccard = np.mean(jaccard_scores)
        results[name] = avg_jaccard
        print(f"  Average Jaccard similarity (top-{k}): {avg_jaccard:.3f}")
        print(f"  (1.0 = no change, 0.0 = completely different recommendations)")

    print(f"\n{'='*55}")
    print("PERTURBATION ANALYSIS SUMMARY")
    print(f"{'='*55}")
    for name, score in results.items():
        stability = "Robust" if score > 0.6 else "Moderate" if score > 0.3 else "Fragile"
        print(f"  {name:38s}: {score:.3f}  ({stability})")

    return results


if __name__ == "__main__":
    results = run_perturbation_analysis(sample_size=50, k=5)