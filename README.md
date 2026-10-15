# Movie Recommendation System

A collaborative-filtering movie recommender built on the MovieLens 1M dataset. It factorizes the user–movie rating matrix with Truncated SVD, predicts ratings for unseen movies, and returns a balanced list of **3 mainstream + 2 niche** recommendations per user, with a penalty for sequels whose prequel the user hasn't watched.

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) (GroupLens Research):

| File | Rows | Columns |
|---|---|---|
| `data/raw/ratings.csv` | 1,000,209 | `userId;movieId;rating;timestamp` |
| `data/raw/movies.csv` | 3,883 | `movieId;title;genres` |
| `data/raw/users.csv` | 6,040 | `userId;gender;age;occupation;zip-code` |

Files are `;`-separated and `latin-1` encoded. See `data/raw/README` for the dataset's license and citation.

## How it works

1. **Preprocessing** (`src/data_loader.py`)
   - Merges ratings with movie titles.
   - Keeps only active users (≥ 50 ratings) and popular movies (≥ 50 ratings) to reduce sparsity.
   - Builds a `userId × title` pivot matrix and applies **mean centering** per user; missing values become 0.
2. **Model** (`src/engine.py`)
   - `TruncatedSVD` with 50 latent factors on the centered matrix.
   - Predicted ratings = `user_features · movie_features + user mean`.
   - The prediction matrix is cached to `models/svd_predictions.pkl` with `joblib`; later runs load it instead of retraining.
3. **Evaluation** (`src/evaluator.py`)
   - RMSE between actual ratings and predictions on the observed (user, movie) pairs.
4. **Recommendation** (`src/recommender.py`)
   - Ranks unseen movies by predicted score.
   - Movies with more than 1,000 ratings count as *Mainstream*, the rest as *Niche*; the final list holds 3 Mainstream and 2 Niche movies.
   - **Sequel awareness:** if a movie has a prequel the user hasn't watched, its score is multiplied by 0.8.

## Project structure

```
├── data/raw/             # MovieLens 1M CSVs
├── models/               # Saved prediction matrix (created on first run)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_recommender.ipynb
│   ├── 03_collaborative_filtering.ipynb
│   └── 04_matrix_factorization.ipynb
├── src/
│   ├── data_loader.py    # Loading, filtering, pivot + mean centering
│   ├── engine.py         # SVD training, saving/loading the model
│   ├── evaluator.py      # RMSE
│   ├── recommender.py    # Hybrid 3 mainstream + 2 niche recommendations
│   └── api/main.py       # End-to-end pipeline entry point
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/ism00efe/movie-recommendation-system.git
cd movie-recommendation-system
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Usage

Run the full pipeline (load → train or load model → RMSE → recommendations) from the project root:

```bash
python src/api/main.py
```

The first run trains the SVD model and saves it to `models/svd_predictions.pkl`; subsequent runs reuse it. The script prints the matrix size, the RMSE score and 5 recommendations for user `42`. Change `user_id` in `src/api/main.py` to get recommendations for another user.

Example of using the modules directly:

```python
from src import load_and_preprocess, create_pivot_matrix, train_model, get_hybrid_recommendations

df = load_and_preprocess("data/raw/ratings.csv", "data/raw/movies.csv")
matrix, matrix_norm = create_pivot_matrix(df)
predictions = train_model(matrix, matrix_norm, n_components=50)

recs = get_hybrid_recommendations(42, predictions, matrix, df, movie_sequels={})
print(recs)
```

## Tech stack

Python · pandas · NumPy · scikit-learn (TruncatedSVD) · joblib · Jupyter

## Future improvements

- [ ] Serve recommendations through a FastAPI endpoint.
- [ ] Evaluate on a held-out test split instead of the training ratings.
- [ ] Build the sequel map automatically instead of a hand-written dictionary.
- [ ] Containerize with Docker.
