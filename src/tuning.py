# ============================================================
# tuning.py — Optuna hyperparameter search for similarity weights
# ============================================================
import numpy as np
import pandas as pd
import optuna
import wandb

from recommender import load_everything, build_similarity_matrices, hybrid_similarity
from evaluation import build_tag_ground_truth, evaluate_recommender


def objective(trial, df, text_sim, numeric_sim, ground_truth):
    """
    Optuna calls this function many times with different
    text_weight values. Whatever this function RETURNS is the
    score Optuna tries to maximize.
    """

    # trial.suggest_float asks Optuna to pick a number in this range
    # for this attempt. Optuna learns from past attempts to pick
    # smarter numbers each time (not random guessing).
    text_weight = trial.suggest_float("text_weight", 0.5, 0.95)
    numeric_weight = 1.0 - text_weight  # must sum to 1.0

    combined = hybrid_similarity(text_sim, numeric_sim,
                                  text_weight=text_weight,
                                  numeric_weight=numeric_weight)

    precision, recall = evaluate_recommender(combined, ground_truth, df,
                                              k=5, sample_size=150)
    
    run = wandb.init(
        project="steamsense-recommender",
        name=f"trial_{trial.number}",
        config={"text_weight": text_weight, "numeric_weight": numeric_weight},
        reinit=True  # allows multiple wandb.init() calls in one script
    )
    run.log({"precision_at_5": precision, "recall_at_5": recall})
    run.finish()

    # We optimize for precision — you could also optimize a
    # blend of precision + recall if you prefer
    return precision


def run_tuning(n_trials=30):
    df, embeddings = load_everything()
    text_sim, numeric_sim = build_similarity_matrices(df, embeddings)
    ground_truth = build_tag_ground_truth(df, min_shared_tags=3)

    # direction="maximize" because higher precision is better
    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, df, text_sim, numeric_sim, ground_truth),
        n_trials=n_trials
    )

    print(f"\nBest text_weight: {study.best_params['text_weight']:.3f}")
    print(f"Best precision@5: {study.best_value:.3f}")

    # Save the best weight so recommender.py can use it later
    best_text_weight = study.best_params["text_weight"]
    with open("models/best_weights.txt", "w") as f:
        f.write(f"text_weight={best_text_weight}\n")
        f.write(f"numeric_weight={1.0 - best_text_weight}\n")

    return study


if __name__ == "__main__":
    study = run_tuning(n_trials=30)