# ============================================================
# data_quality.py — Advanced Data Quality Module (v2)
# ============================================================
# This version goes beyond basic null/duplicate checks.
# It finds SEMANTIC problems that hurt your recommender:
#   1. Semantic duplicates (same game, different app IDs)
#   2. Genre imbalance (too many Action games)
#   3. Tag noise (irrelevant or adult tags)
#   4. Boilerplate descriptions (useless for embeddings)
#   5. Metacritic score bias (70% are -1, can't use raw)
#   6. Description richness scoring (quality score per game)
# ============================================================

import pandas as pd
import numpy as np
import json
import os
from collections import Counter


# ============================================================
# PART 1: Load data
# ============================================================

def load_raw_data(path="data/raw/steam_games_raw.csv"):
    print("Loading raw data...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns\n")

    df["price_eur"]        = pd.to_numeric(df["price_eur"], errors="coerce")
    df["metacritic_score"] = pd.to_numeric(df["metacritic_score"], errors="coerce")
    df["app_id"]           = pd.to_numeric(df["app_id"], errors="coerce")

    text_cols = ["name", "short_description", "detailed_description",
                 "genres", "categories", "tags", "review_score",
                 "developers", "publishers", "platforms", "release_date"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


# ============================================================
# PART 2: Basic quality metrics (same as before, now a function)
# ============================================================

def measure_basic_quality(df, label="raw"):
    print(f"\n{'='*55}")
    print(f"BASIC QUALITY METRICS — {label.upper()}")
    print(f"{'='*55}")

    metrics = {"label": label, "total_rows": len(df)}

    # Missing values
    print("\n[1] Missing / empty value rates:")
    missing = {}
    for col in df.columns:
        if df[col].dtype == object:
            empty = (df[col].isna() | (df[col].str.strip() == "")).sum()
        else:
            empty = df[col].isna().sum()
        pct = round(empty / len(df) * 100, 1)
        missing[col] = {"count": int(empty), "percent": pct}
        if empty > 0:
            print(f"  {col:30s}: {empty:4d} ({pct:.1f}%)")

    metrics["missing"] = missing

    # Duplicates
    dupes = df.duplicated(subset=["app_id"]).sum()
    print(f"\n[2] Exact duplicate app_ids: {dupes}")
    metrics["exact_duplicates"] = int(dupes)

    # Description length
    lengths = df["short_description"].str.len()
    short = (lengths < 20).sum()
    print(f"\n[3] Descriptions: avg={lengths.mean():.0f} chars, min={lengths.min()}, under 20 chars: {short}")
    metrics["description"] = {"avg_len": round(float(lengths.mean()),1), "short_count": int(short)}

    # Price
    prices = df["price_eur"].dropna()
    print(f"\n[4] Price: min=€{prices.min():.2f}, max=€{prices.max():.2f}, avg=€{prices.mean():.2f}")
    print(f"    Free games: {(prices == 0).sum()}, Negative: {(prices < 0).sum()}, Over €100: {(prices > 100).sum()}")
    metrics["price"] = {"min": float(prices.min()), "max": float(prices.max()), "mean": round(float(prices.mean()),2)}

    # Genre distribution
    all_genres = []
    for g in df["genres"]:
        if g:
            for genre in g.split(","):
                all_genres.append(genre.strip())
    genre_counts = Counter(all_genres)
    print(f"\n[5] Top genres:")
    for genre, count in genre_counts.most_common(8):
        bar = "█" * (count // 10)
        print(f"    {genre:22s}: {count:4d}  {bar}")
    metrics["genres"] = dict(genre_counts.most_common(15))

    return metrics


# ============================================================
# PART 3: ADVANCED quality checks — the ones that matter
# ============================================================

def advanced_quality_checks(df):
    """
    Runs 5 advanced checks that basic cleaning misses.
    Returns a report dict and a list of flagged app_ids per issue.
    """

    print(f"\n{'='*55}")
    print("ADVANCED QUALITY CHECKS")
    print(f"{'='*55}")

    report = {}
    flags = {}   # app_id → list of issues found

    # ---- Check 1: Semantic duplicates ----
    # Two different app_ids can be the same game (demo + full version,
    # or regional editions like "Game (EU)" and "Game (NA)").
    # We detect these by finding games with nearly identical names.
    # Why this matters: your recommender would recommend "Hades Demo"
    # when the user picks "Hades" — that is useless.

    print("\n[A] Semantic duplicate detection (same name, different IDs):")

    # Normalize names: lowercase, remove special chars, strip edition words
    import re
    def normalize_name(name):
        name = name.lower()
        name = re.sub(r"[^a-z0-9 ]", "", name)
        # Remove common edition suffixes
        for suffix in ["demo", "beta", "trial", "free edition", "lite",
                       "prologue", "chapter 1", "episode 1", "early access"]:
            name = name.replace(suffix, "").strip()
        return name.strip()

    df["_norm_name"] = df["name"].apply(normalize_name)

    # Find groups where the normalized name appears more than once
    name_counts = df["_norm_name"].value_counts()
    duplicate_names = name_counts[name_counts > 1].index.tolist()

    semantic_dupes = df[df["_norm_name"].isin(duplicate_names)][["app_id", "name", "_norm_name"]]
    print(f"  Found {len(semantic_dupes)} games that are likely semantic duplicates:")
    for _, row in semantic_dupes.head(10).iterrows():
        print(f"    app_id={row['app_id']}: '{row['name']}'  →  normalized: '{row['_norm_name']}'")

    report["semantic_duplicates"] = {
        "count": len(semantic_dupes),
        "examples": semantic_dupes["name"].head(10).tolist()
    }

    # Flag: for each duplicate group, keep the one with more reviews/better data
    # Strategy: keep the highest app_id (usually the newer/main release)
    dupes_to_remove = []
    for norm_name in duplicate_names:
        group = df[df["_norm_name"] == norm_name].sort_values("app_id", ascending=False)
        # Keep the first (highest app_id), flag the rest
        dupes_to_remove.extend(group.iloc[1:]["app_id"].tolist())

    flags["semantic_duplicates"] = dupes_to_remove
    print(f"  Will remove {len(dupes_to_remove)} semantic duplicate rows")

    # ---- Check 2: Boilerplate / low-quality descriptions ----
    # These are descriptions that contain no useful semantic content.
    # Examples:
    #   "Coming Soon"
    #   "This game is currently in development"
    #   A description that is just the game's URL
    # Why this matters: embedding "Coming Soon" gives a near-zero vector
    # that will randomly match other games.

    print("\n[B] Boilerplate description detection:")

    boilerplate_phrases = [
        "coming soon",
        "currently in development",
        "stay tuned",
        "wishlist now",
        "this game will be",
        "no description available",
        "tba",
        "to be announced",
        "work in progress",
        "early access game",   # just the label, no actual description
    ]

    def is_boilerplate(desc):
        if not desc:
            return True
        desc_lower = desc.lower().strip()
        # Too short
        if len(desc_lower) < 30:
            return True
        # Matches boilerplate patterns
        for phrase in boilerplate_phrases:
            if phrase in desc_lower and len(desc_lower) < 80:
                # Only flag if the description is MOSTLY this phrase
                # A 500-word description mentioning "coming soon" is fine
                return True
        # Description is just a URL
        if desc_lower.startswith("http") or desc_lower.startswith("www"):
            return True
        return False

    df["_is_boilerplate"] = df["short_description"].apply(is_boilerplate)
    boilerplate_count = df["_is_boilerplate"].sum()

    print(f"  Found {boilerplate_count} games with boilerplate/useless descriptions:")
    bad_descs = df[df["_is_boilerplate"]][["app_id", "name", "short_description"]].head(8)
    for _, row in bad_descs.iterrows():
        print(f"    '{row['name']}': \"{row['short_description'][:60]}\"")

    report["boilerplate_descriptions"] = {
        "count": int(boilerplate_count),
        "examples": bad_descs["name"].tolist()
    }
    flags["boilerplate"] = df[df["_is_boilerplate"]]["app_id"].tolist()

    # ---- Check 3: Tag noise ----
    # User tags on Steam include some that are completely useless
    # for a game recommender (they describe content warnings, not game style).
    # If "Nudity" is a tag, the embedding includes that concept
    # and will match games based on nudity rather than genre/gameplay.

    print("\n[C] Tag noise analysis:")

    # Tags that carry no useful recommendation signal
    noise_tags = {
        "Nudity", "Sexual Content", "Adult Only", "NSFW",
        "Gore", "Violent", "Mature", "18+",
        "Free to Play",   # this is a business model, not a game type
        "Early Access",   # same — already in genres column
        "Downloadable Content",  # this is DLC, not a game tag
    }

    def clean_tags(tags_str):
        """Remove noise tags from a game's tag list."""
        if not tags_str:
            return ""
        tags = [t.strip() for t in tags_str.split(",")]
        clean = [t for t in tags if t not in noise_tags]
        return ", ".join(clean)

    def count_noise_tags(tags_str):
        """Count how many noise tags a game has."""
        if not tags_str:
            return 0
        tags = [t.strip() for t in tags_str.split(",")]
        return sum(1 for t in tags if t in noise_tags)

    df["_noise_tag_count"] = df["tags"].apply(count_noise_tags)
    games_with_noise = (df["_noise_tag_count"] > 0).sum()
    total_noise = df["_noise_tag_count"].sum()

    print(f"  Games with at least 1 noise tag: {games_with_noise}")
    print(f"  Total noise tags to remove: {total_noise}")
    print(f"  Noise tags being removed: {noise_tags}")

    report["tag_noise"] = {
        "games_affected": int(games_with_noise),
        "total_tags_removed": int(total_noise),
        "noise_tag_list": list(noise_tags)
    }
    # We don't remove these GAMES — we just clean their tags
    # This is a fix, not a removal

    # ---- Check 4: Metacritic score bias analysis ----
    # 70% of games have metacritic_score = -1 (we set it to -1 for missing).
    # If we use this column directly in numerical similarity,
    # -1 will dominate and all "unreviewed" games will cluster together.
    # Solution: create a separate boolean flag "has_metacritic_score"
    # and normalize only the actual scores to 0-1 range.

    print("\n[D] Metacritic score bias:")

    has_score = df["metacritic_score"].notna().sum()
    no_score  = df["metacritic_score"].isna().sum()
    pct_missing = no_score / len(df) * 100

    print(f"  Games with Metacritic score: {has_score} ({100-pct_missing:.1f}%)")
    print(f"  Games WITHOUT score (-1):    {no_score} ({pct_missing:.1f}%)")

    if has_score > 0:
        real_scores = df[df["metacritic_score"] != -1]["metacritic_score"]
        print(f"  Score range (where available): {real_scores.min():.0f} – {real_scores.max():.0f}")
        print(f"  Average score (where available): {real_scores.mean():.1f}")

    print(f"  FIX: Will add 'has_metacritic' flag column + normalize scores to 0-1")

    report["metacritic_bias"] = {
        "has_score_count": int(has_score),
        "missing_count": int(no_score),
        "missing_percent": round(pct_missing, 1)
    }

    # ---- Check 5: Description richness score ----
    # Not all descriptions are equal. A 20-character description
    # gives terrible embeddings. We score each description 0-100
    # so we can show this metric in our presentation.
    # Why: this demonstrates you thought about EMBEDDING QUALITY,
    # not just data completeness.

    print("\n[E] Description richness scoring (0–100):")

    def richness_score(row):
        """
        Scores how useful a game's text data will be for embeddings.
        Returns 0-100.
        """
        score = 0

        # Short description length (max 40 points)
        desc_len = len(str(row.get("short_description", "")))
        score += min(40, desc_len / 5)
        # A 200-char description gets 40 points (capped)

        # Has tags (max 30 points)
        tags = str(row.get("tags", ""))
        tag_count = len([t for t in tags.split(",") if t.strip()])
        score += min(30, tag_count * 3)
        # 10 tags = 30 points (capped)

        # Has genres (max 15 points)
        genres = str(row.get("genres", ""))
        if genres.strip():
            score += 15

        # Has metacritic score (max 15 points)
        meta = row.get("metacritic_score", -1)
        if meta != -1 and pd.notna(meta):
            score += 15

        return round(min(100, score))

    df["richness_score"] = df.apply(richness_score, axis=1)

    print(f"  Average richness score: {df['richness_score'].mean():.1f}/100")
    print(f"  Games with score < 40 (poor): {(df['richness_score'] < 40).sum()}")
    print(f"  Games with score ≥ 80 (great): {(df['richness_score'] >= 80).sum()}")

    # Show examples at each tier
    poor  = df[df["richness_score"] < 40][["name","richness_score"]].head(3)
    great = df[df["richness_score"] >= 80][["name","richness_score"]].head(3)
    print(f"\n  Examples — poor richness (<40):")
    for _, r in poor.iterrows():
        print(f"    {r['name']}: {r['richness_score']}/100")
    print(f"  Examples — great richness (≥80):")
    for _, r in great.iterrows():
        print(f"    {r['name']}: {r['richness_score']}/100")

    report["richness_scores"] = {
        "mean": round(float(df["richness_score"].mean()), 1),
        "poor_count": int((df["richness_score"] < 40).sum()),
        "great_count": int((df["richness_score"] >= 80).sum())
    }

    # Clean up temp columns before returning
    df.drop(columns=["_norm_name", "_is_boilerplate", "_noise_tag_count"],
            inplace=True, errors="ignore")

    return df, report, flags


# ============================================================
# PART 4: Apply all fixes
# ============================================================

def apply_all_fixes(df, flags):
    """
    Applies every fix identified in advanced_quality_checks.
    Returns cleaned DataFrame and a log of what changed.
    """

    print(f"\n{'='*55}")
    print("APPLYING FIXES")
    print(f"{'='*55}")

    log = []
    original_count = len(df)

    import re

    # Fix 1: Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["app_id"], keep="first")
    removed = before - len(df)
    log.append(f"Removed {removed} exact duplicate app_ids")
    print(f"\n[1] Removed {removed} exact duplicates")

    # Fix 2: Remove semantic duplicates
    before = len(df)
    if flags.get("semantic_duplicates"):
        df = df[~df["app_id"].isin(flags["semantic_duplicates"])]
    removed = before - len(df)
    log.append(f"Removed {removed} semantic duplicates (same game, different editions)")
    print(f"[2] Removed {removed} semantic duplicates")

    # Fix 3: Remove boilerplate descriptions
    before = len(df)
    if flags.get("boilerplate"):
        df = df[~df["app_id"].isin(flags["boilerplate"])]
    removed = before - len(df)
    log.append(f"Removed {removed} games with boilerplate/useless descriptions")
    print(f"[3] Removed {removed} boilerplate description games")

    # Fix 4: Clean noise tags
    noise_tags = {
        "Nudity", "Sexual Content", "Adult Only", "NSFW",
        "Gore", "Violent", "Mature", "18+",
        "Free to Play", "Early Access", "Downloadable Content"
    }

    def clean_tags(tags_str):
        if not tags_str:
            return ""
        tags = [t.strip() for t in tags_str.split(",")]
        return ", ".join([t for t in tags if t not in noise_tags])

    df["tags"] = df["tags"].apply(clean_tags)
    log.append(f"Cleaned noise tags from all games: {noise_tags}")
    print(f"[4] Cleaned noise tags from all games")

    # Fix 5: Normalize metacritic score + add flag column
    df["has_metacritic"] = df["metacritic_score"].notna().astype(int)
    # has_metacritic: 1 if the game has a real score, 0 if not

    # Normalize the actual scores to 0-1 range for use in similarity
    # Games without a score get 0.5 (neutral, not bad, not good)
    def normalize_metacritic(row):
        if row["has_metacritic"] == 0:
            return 0.5  # neutral default
        return row["metacritic_score"] / 100.0
        # 0 = terrible, 1.0 = perfect

    df["metacritic_normalized"] = df.apply(normalize_metacritic, axis=1)
    log.append("Added has_metacritic (0/1) and metacritic_normalized (0–1) columns")
    print(f"[5] Added metacritic_normalized column (0–1 scale, 0.5 for missing)")

    # Fix 6: Fix prices
    df["price_eur"] = pd.to_numeric(df["price_eur"], errors="coerce").fillna(0)
    df.loc[df["price_eur"] < 0, "price_eur"] = 0.0
    df.loc[df["price_eur"] > 100, "price_eur"] = 80.0
    log.append("Fixed negative prices and capped extreme prices at €80")
    print(f"[6] Fixed price outliers")

    # Fix 7: Normalize review scores
    review_mapping = {
        "overwhelmingly positive": "Overwhelmingly Positive",
        "very positive":           "Very Positive",
        "mostly positive":         "Mostly Positive",
        "mixed":                   "Mixed",
        "mostly negative":         "Mostly Negative",
        "very negative":           "Very Negative",
        "overwhelmingly negative": "Overwhelmingly Negative",
        "positive":                "Positive",
        "negative":                "Negative",
    }
    df["review_score"] = df["review_score"].apply(
        lambda s: review_mapping.get(str(s).lower().strip(), s) if s else "Unknown"
    )
    log.append("Normalized review_score labels")
    print(f"[7] Normalized review score labels")

    # Fix 8: Strip whitespace
    for col in ["name", "genres", "tags", "categories", "developers",
                "publishers", "review_score"]:
        if col in df.columns:
            df[col] = df[col].str.strip()
    log.append("Stripped whitespace from all text columns")
    print(f"[8] Stripped whitespace")

    # Fix 9: Create combined_text for embeddings
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

    df["combined_text"] = df.apply(build_combined_text, axis=1)
    log.append("Created combined_text column for embeddings")
    print(f"[9] Created combined_text column")

    # Final stats
    df = df.reset_index(drop=True)
    total_removed = original_count - len(df)

    print(f"\n  Original: {original_count} rows")
    print(f"  Removed:  {total_removed} rows")
    print(f"  Final:    {len(df)} rows")

    return df, log


# ============================================================
# PART 5: Save everything
# ============================================================

def save_outputs(df, metrics_before, metrics_after, adv_report, log,
                 clean_path="data/processed/steam_games_clean.csv",
                 report_path="data/processed/quality_report.json"):

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(clean_path, index=False)
    print(f"\nSaved cleaned data → {clean_path}")

    full_report = {
        "basic_before": metrics_before,
        "basic_after":  metrics_after,
        "advanced_checks": adv_report,
        "cleaning_log": log,
        "summary": {
            "rows_before": metrics_before["total_rows"],
            "rows_after":  metrics_after["total_rows"],
            "rows_removed": metrics_before["total_rows"] - metrics_after["total_rows"],
            "new_columns_added": ["has_metacritic", "metacritic_normalized",
                                  "combined_text", "richness_score"],
            "columns_final": list(df.columns)
        }
    }

    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Saved quality report → {report_path}")

    print(f"\n{'='*55}")
    print("FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"Rows before : {full_report['summary']['rows_before']}")
    print(f"Rows removed: {full_report['summary']['rows_removed']}")
    print(f"Rows after  : {full_report['summary']['rows_after']}")
    print(f"\nCleaning steps:")
    for step in log:
        print(f"  • {step}")
    print(f"\nNew columns added: {full_report['summary']['new_columns_added']}")
    print(f"\nFinal shape: {df.shape[0]} games × {df.shape[1]} columns")
    print(f"{'='*55}")


# ============================================================
# PART 6: Run everything
# ============================================================

def run_quality_pipeline():
    df_raw = load_raw_data("data/raw/steam_games_raw.csv")

    metrics_before = measure_basic_quality(df_raw, label="raw")

    df_raw, adv_report, flags = advanced_quality_checks(df_raw)

    df_clean, log = apply_all_fixes(df_raw, flags)

    metrics_after = measure_basic_quality(df_clean, label="clean")

    save_outputs(df_clean, metrics_before, metrics_after, adv_report, log)

    print("\nDay 2 complete. Ready for Day 3.")
    return df_clean


if __name__ == "__main__":
    df_clean = run_quality_pipeline()