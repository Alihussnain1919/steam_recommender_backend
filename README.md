# Steam Recommender System


## 🛠️ Step-by-Step Local Setup & Execution

Follow these precise steps to pull down the repository, set up your localized isolated workspace, and start up the server.

### 1. Initialize and Activate the Virtual Environment (`.venv`)
Before installing anything, always establish your isolated virtual environment folder to ensure libraries do not clash with your system.

```bash
# Move inside the project folder
cd steam_recommender

# Create a fresh local virtual environment
python -m venv .venv

# Activate the virtual environment:
# On macOS / Linux:
source .venv/bin/activate

# On Windows (Command Prompt):
# .venv\Scripts\activate.bat
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1


2. Install Packages from Scratch
Once your terminal shows the active (.venv) prefix, run pip to load all development, vector processing, and machine learning dependencies:
Bash
pip install -r src/requirements.txt
3. Run the Backend API Service Locally
To run the server locally on your laptop with live hot-reloading (the server resets every time you change code), run this:
Bash
uvicorn src.api:app --reload --port 8000
Your local microservice will now be active and listening at: http://localhost:8000


## Project Architecture Flowchart

## 📦 Project Libraries & Tools Breakdown

| Library | What it does | Why you need it |
| :--- | :--- | :--- |
| **`requests`** | Makes HTTP requests — like your browser visiting a URL, but in Python | To call the Steam API |
| **`beautifulsoup4`** | Reads HTML and lets you find specific parts of it | To scrape individual Steam pages |
| **`pandas`** | Handles tables of data (like Excel but in Python) | To store and manipulate your game data |
| **`numpy`** | Math library for numbers and arrays | For embeddings and similarity math |
| **`scikit-learn`** | Machine learning toolkit | For cosine similarity, evaluation |
| **`sentence-transformers`** | Pre-trained language model that converts text to vectors | For game description embeddings |
| **`optuna`** | Automatically finds best hyperparameters | For tuning your recommender |
| **`wandb`** | Experiment tracking dashboard | For logging results |
| **`streamlit`** | Builds web apps with pure Python | For your frontend |
| **`tqdm`** | Shows a progress bar in terminal | So you can see scraping progress |



# Steam Recommendation System Backend

This project serves as a high-performance content-based recommendation engine for Steam games. Powered by **FastAPI** and **Sentence-Transformers**, the backend uses semantic textual data (like descriptions and genres) to pre-compute vector embeddings for thousands of games. It exposes an active microservice API that instantly provides highly accurate game suggestions to our React frontend based on a user's selected game, budget preferences, or favorite genres.

---




