# ============================================================
# scraper.py — Steam Game Data Collector (Final Version)
# ============================================================
# FLOW:
#   1. Scrape Steam search pages with BeautifulSoup → get App IDs
#   2. Call Steam App Details API per ID → get structured data
#   3. Scrape the individual store page → get user tags
#   4. Save everything to a CSV
# ============================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from tqdm import tqdm


# ============================================================
# PART 1: Scrape App IDs from Steam search pages
# ============================================================
# WHY THIS INSTEAD OF THE API:
#   The GetAppList API endpoint often returns empty or rate-limits you.
#   Steam's search page is publicly available and always works.
#   Each result row has a data-ds-appid attribute = the app ID we need.
#   We paginate through search pages until we have enough IDs.

def get_app_ids_from_search(max_ids=1300):
    """
    Scrapes Steam search pages to collect app IDs.
    We collect max_ids (more than 1000) to have a buffer —
    because some IDs will fail the API check or not be real games.

    Returns: list of integer app IDs
    """

    headers = {
        # We pretend to be a browser so Steam doesn't block us
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    app_ids = []
    page = 1

    print("Step 1: Collecting app IDs from Steam search pages...")

    while len(app_ids) < max_ids:
        # Steam search page URL — page parameter controls pagination
        # ?type=game filters to actual games only (no DLC, music, videos)
        url = f"https://store.steampowered.com/search/?type=game&page={page}"

        try:
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code != 200:
                print(f"  Page {page} returned status {response.status_code}. Stopping.")
                break

            soup = BeautifulSoup(response.text, "html.parser")

            # Each game on the search results page is an <a> tag with class "search_result_row"
            # It contains: data-ds-appid="570" (for Dota 2)
            game_rows = soup.find_all("a", class_="search_result_row")

            if not game_rows:
                print(f"  No games found on page {page}. Stopping.")
                break

            for row in game_rows:
                # data-ds-appid is the Steam App ID — this is what we need
                app_id_str = row.get("data-ds-appid")
                if app_id_str:
                    try:
                        app_ids.append(int(app_id_str))
                    except ValueError:
                        continue  # Skip if it's not a valid number

            # Remove duplicates while preserving order
            seen = set()
            unique_ids = []
            for id in app_ids:
                if id not in seen:
                    seen.add(id)
                    unique_ids.append(id)
            app_ids = unique_ids

            print(f"  Page {page}: {len(app_ids)} unique IDs collected so far...")

            page += 1

            # Wait 2 seconds between page requests — be a polite scraper
            time.sleep(2)

        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break

    app_ids = app_ids[:max_ids]
    print(f"  Done. Collected {len(app_ids)} app IDs total.\n")
    return app_ids


# ============================================================
# PART 2: Get detailed data for one game from the Steam API
# ============================================================
# WHY USE THE API HERE (not scraping):
#   The API returns clean, structured JSON data — price, genres,
#   categories, description, platforms, etc. Much easier to parse
#   than scraping all that from HTML.
#   We only scrape the store page for TAGS (which the API doesn't give).

def get_game_details_from_api(app_id):
    """
    Calls Steam's App Details API for a single game.
    Returns a dict of game data, or None if the game is invalid.

    app_id: integer, e.g. 730 for CS2
    """

    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=de&l=en"
    # cc=de  → prices in Euros
    # l=en   → descriptions in English
    # appids → the specific game

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)

        # Check if response is actually JSON (sometimes Steam returns empty)
        if not response.text.strip():
            return None

        data = response.json()

        # Steam wraps the response with the app_id as the key (as a string)
        # e.g. { "730": { "success": true, "data": { ... } } }
        app_data = data.get(str(app_id), {})

        if not app_data.get("success", False):
            return None  # App doesn't exist or was removed

        details = app_data.get("data", {})

        # We only want apps of type "game"
        # Other types: "dlc", "music", "video", "demo", "tool"
        if details.get("type", "") != "game":
            return None

        # ---- Extract all fields ----

        name = details.get("name", "Unknown")
        short_description = details.get("short_description", "")
        detailed_description = details.get("detailed_description", "")

        # Genres: API returns [{"id":"1","description":"Action"}, ...]
        genres_raw = details.get("genres", [])
        genres = ", ".join([g["description"] for g in genres_raw])

        # Categories: [{"id":2,"description":"Single-player"}, ...]
        categories_raw = details.get("categories", [])
        categories = ", ".join([c["description"] for c in categories_raw])

        # Price in Euros (API returns cents, so divide by 100)
        price_overview = details.get("price_overview", {})
        if price_overview:
            price_eur = price_overview.get("final", 0) / 100
        else:
            price_eur = 0.0  # Free-to-play games have no price_overview

        is_free = details.get("is_free", False)

        # Metacritic score (integer 0-100, or None if not reviewed)
        metacritic = details.get("metacritic", {})
        metacritic_score = metacritic.get("score", None)

        # Steam user review score (this comes from the store page, not API directly)
        # We'll add this in Part 3 via BeautifulSoup

        release_date = details.get("release_date", {}).get("date", "Unknown")
        developers = ", ".join(details.get("developers", []))
        publishers = ", ".join(details.get("publishers", []))

        # Platforms: {"windows": true, "mac": false, "linux": true}
        platforms = details.get("platforms", {})
        platform_list = [p for p, supported in platforms.items() if supported]
        platforms_str = ", ".join(platform_list)

        # Header image URL (we'll use this in the Streamlit frontend)
        header_image = details.get("header_image", "")

        return {
            "app_id": app_id,
            "name": name,
            "short_description": short_description,
            "detailed_description": detailed_description,
            "genres": genres,
            "categories": categories,
            "price_eur": price_eur,
            "is_free": is_free,
            "metacritic_score": metacritic_score,
            "release_date": release_date,
            "developers": developers,
            "publishers": publishers,
            "platforms": platforms_str,
            "header_image": header_image,
        }

    except Exception as e:
        # Don't crash — just skip this game and move on
        return None


# ============================================================
# PART 3: Scrape user tags and review score from store page
# ============================================================
# WHY SCRAPE THIS:
#   The API does NOT return:
#     - User-defined tags (e.g. "Roguelike", "Dark Fantasy", "Chill")
#     - The overall review sentiment ("Very Positive", "Overwhelmingly Positive")
#   These are only on the HTML store page.
#   Tags are especially important for our recommender system —
#   they carry rich semantic meaning about game feel/style.

def get_tags_and_rating_from_page(app_id):
    """
    Scrapes the Steam store page for a game.
    Returns: (tags_string, review_score_string)

    tags_string       : e.g. "Roguelike, Indie, Difficult, Turn-Based"
    review_score_string: e.g. "Very Positive" or "" if not found
    """

    url = f"https://store.steampowered.com/app/{app_id}/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        # These cookies bypass the age gate for mature games
        # Without this, Steam shows a "Enter your birthdate" page instead
        "Cookie": "birthtime=470700001; lastagecheckage=1-January-1985; wants_mature_content=1",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        # --- Scrape user tags ---
        # HTML structure on the page:
        # <a class="app_tag" href="/search/?tags=1625">Roguelike</a>
        tag_elements = soup.find_all("a", class_="app_tag")
        tags = [tag.text.strip() for tag in tag_elements[:15]]
        # We keep max 15 tags — more than that is noise
        tags_string = ", ".join(tags)

        # --- Scrape review score ---
        # The overall review is in a <span> with class "game_review_summary"
        # e.g. <span class="game_review_summary positive">Very Positive</span>
        review_element = soup.find("span", class_="game_review_summary")
        if review_element:
            review_score = review_element.text.strip()
        else:
            review_score = ""

        return tags_string, review_score

    except Exception:
        return "", ""  # Scraping failed — return empty, not a fatal error


# ============================================================
# PART 4: Clean HTML from text fields
# ============================================================
# WHY: API descriptions contain raw HTML like:
#   "<p>Explore a <strong>huge</strong> world...</p><br>"
# We want plain text: "Explore a huge world..."

def clean_html(html_text):
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    # get_text() extracts only visible text, strips all HTML tags
    # separator=" " puts a space between text blocks
    return soup.get_text(separator=" ").strip()


# ============================================================
# PART 5: Main pipeline
# ============================================================

def scrape_steam_games(max_games=1000, save_path="data/raw/steam_games_raw.csv"):
    """
    Full pipeline:
      1. Collect ~1300 app IDs from search pages (buffer for failures)
      2. For each ID: get API details + scrape tags
      3. Save to CSV every 50 games (so crashes don't lose progress)
      4. Stop when we have max_games valid games

    max_games : target number of valid games to collect
    save_path : where to save the output CSV
    """

    os.makedirs("data/raw", exist_ok=True)

    # Resume from existing file if scraping was interrupted
    if os.path.exists(save_path):
        print(f"Found existing data at {save_path}. Loading to resume...")
        existing_df = pd.read_csv(save_path)
        already_collected_ids = set(existing_df["app_id"].tolist())
        games = existing_df.to_dict("records")
        print(f"Already have {len(games)} games. Resuming...\n")
    else:
        already_collected_ids = set()
        games = []

    # Collect more IDs than we need (buffer for API failures)
    # Typically ~20-30% of IDs fail (not real games, API errors, etc.)
    target_ids = get_app_ids_from_search(max_ids=max_games + 400)

    # Remove IDs we already processed
    target_ids = [id for id in target_ids if id not in already_collected_ids]
    print(f"IDs to process: {len(target_ids)}\n")

    # Main loop
    for app_id in tqdm(target_ids, desc="Collecting games"):

        # Stop if we reached our target
        if len(games) >= max_games:
            print(f"\nReached target of {max_games} games. Stopping.")
            break

        # Step A: Get structured data from API
        game_data = get_game_details_from_api(app_id)

        if game_data is None:
            # Not a valid game — skip
            time.sleep(0.5)
            continue

        # Step B: Get tags and review score from store page
        tags, review_score = get_tags_and_rating_from_page(app_id)
        game_data["tags"] = tags
        game_data["review_score"] = review_score

        # Step C: Clean HTML from text fields
        game_data["short_description"] = clean_html(game_data["short_description"])
        game_data["detailed_description"] = clean_html(game_data["detailed_description"])

        games.append(game_data)

        # Save every 50 games — protects against crashes
        if len(games) % 50 == 0:
            pd.DataFrame(games).to_csv(save_path, index=False)
            print(f"\n  Saved checkpoint: {len(games)} games collected.")

        # Wait 1.5 seconds between games — Steam rate limit protection
        # 1.5s × 1000 games ≈ 25 minutes total
        time.sleep(1.5)

    # Final save
    df = pd.DataFrame(games)
    df.to_csv(save_path, index=False)

    print(f"\n{'='*50}")
    print(f"DONE! Collected {len(df)} valid games.")
    print(f"Saved to: {save_path}")
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nSample of collected data:")
    print(df[["name", "genres", "price_eur", "review_score", "tags"]].head(10).to_string())

    return df


# ============================================================
# Run it
# ============================================================
if __name__ == "__main__":
    df = scrape_steam_games(max_games=1000)