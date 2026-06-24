# ============================================================
# embeddings.py — Generate vector embeddings for every game
# ============================================================
# Input:  data/processed/steam_games_clean.csv
# Output: models/embeddings.npy        (the vectors)
#         models/embeddings_ids.csv    (app_id order, so vectors
#                                        can be matched back to games)
# ============================================================

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import time


def generate_embeddings(
    input_path="data/processed/steam_games_clean.csv",
    output_vectors="models/embeddings.npy",
    output_ids="models/embeddings_ids.csv"
):
    print("Loading cleaned data...")
    df = pd.read_csv(input_path)
    df["combined_text"] = df["combined_text"].fillna("")
    print(f"  {len(df)} games loaded")

    # Load the pretrained model
    # This downloads ~80MB the first time, then caches locally
    print("\nLoading sentence-transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Why this model: small (80MB), fast on CPU, 384-dim output,
    # strong general-purpose semantic quality — ideal for a
    # student project without a GPU.

    print("Generating embeddings for all games...")
    start = time.time()

    texts = df["combined_text"].tolist()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32
        # batch_size: how many texts the model processes at once
        # 32 is a safe default for CPU; higher uses more memory
    )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f} seconds")
    print(f"Embedding matrix shape: {embeddings.shape}")
    # shape = (999, 384) → 999 games, each represented by 384 numbers

    os.makedirs("models", exist_ok=True)

    # Save the vectors as a binary numpy file (fast to load later)
    np.save(output_vectors, embeddings)

    # Save the app_id order — CRITICAL: row i in embeddings.npy
    # corresponds to row i in this CSV. We need this mapping
    # because numpy arrays don't carry column names.
    df[["app_id", "name"]].to_csv(output_ids, index=False)

    print(f"\nSaved vectors to: {output_vectors}")
    print(f"Saved ID mapping to: {output_ids}")

    return embeddings, df


if __name__ == "__main__":
    embeddings, df = generate_embeddings()