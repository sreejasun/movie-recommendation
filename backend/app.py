from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


class RecommenderService:
    """Fast startup recommender based on user/item bias estimates from train data."""

    def __init__(self) -> None:
        self.mu: float = 0.0
        self.user_bias: dict[int, float] = {}
        self.item_bias: dict[int, float] = {}
        self.movies: pd.DataFrame = pd.DataFrame()
        self.all_movie_ids: np.ndarray = np.array([], dtype=int)
        self.user_rated: dict[int, set[int]] = {}
        self.metrics_df: pd.DataFrame = pd.DataFrame()
        self.ready: bool = False
        self._load()

    def _load_train(self) -> pd.DataFrame:
        train_path = DATA_DIR / "train.parquet"
        if train_path.exists():
            return pd.read_parquet(train_path)

        rating_path = DATA_DIR / "rating.csv"
        if not rating_path.exists():
            raise FileNotFoundError(
                "Missing data/train.parquet and data/rating.csv. "
                "Run pipeline first or place dataset files under data/."
            )

        # Fallback for first-time setup; uses full CSV.
        ratings = pd.read_csv(rating_path, usecols=["userId", "movieId", "rating"])
        ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        return ratings

    def _load_metrics(self) -> pd.DataFrame:
        metrics_path = RESULTS_DIR / "all_model_results.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path, index_col=0)
            return df
        return pd.DataFrame(columns=["RMSE", "MAE"])

    def _load(self) -> None:
        train = self._load_train()
        movies_path = DATA_DIR / "movie.csv"
        if not movies_path.exists():
            raise FileNotFoundError("Missing data/movie.csv")

        self.movies = pd.read_csv(movies_path, usecols=["movieId", "title"]).copy()
        self.movies["movieId"] = self.movies["movieId"].astype(int)
        self.movies = self.movies.drop_duplicates(subset=["movieId"])
        self.all_movie_ids = self.movies["movieId"].to_numpy(dtype=int)

        self.mu = float(train["rating"].mean())
        user_mean = train.groupby("userId")["rating"].mean()
        item_mean = train.groupby("movieId")["rating"].mean()

        self.user_bias = (user_mean - self.mu).to_dict()
        self.item_bias = (item_mean - self.mu).to_dict()

        rated_pairs = train.groupby("userId")["movieId"].apply(lambda s: set(s.astype(int)))
        self.user_rated = rated_pairs.to_dict()
        self.metrics_df = self._load_metrics()
        self.ready = True

    def _predict_for_user(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        bu = self.user_bias.get(user_id, 0.0)
        bi = np.array([self.item_bias.get(int(mid), 0.0) for mid in movie_ids], dtype=float)
        pred = self.mu + bu + bi
        return np.clip(pred, 0.5, 5.0)

    def recommend(self, user_id: int, k: int = 10) -> list[dict[str, Any]]:
        if user_id not in self.user_rated:
            raise KeyError(f"user_id={user_id} not found")

        seen = self.user_rated[user_id]
        candidates = np.array([mid for mid in self.all_movie_ids if int(mid) not in seen], dtype=int)
        if len(candidates) == 0:
            return []

        scores = self._predict_for_user(user_id, candidates)
        top_idx = np.argsort(scores)[::-1][:k]
        top_ids = candidates[top_idx]
        top_scores = scores[top_idx]

        title_map = dict(zip(self.movies["movieId"].astype(int), self.movies["title"]))
        return [
            {
                "movie_id": int(mid),
                "title": title_map.get(int(mid), str(mid)),
                "predicted_rating": float(score),
            }
            for mid, score in zip(top_ids, top_scores)
        ]

    def metrics_payload(self) -> dict[str, Any]:
        if self.metrics_df.empty:
            return {"rmse": None, "mae": None, "best_model": None, "all_models": []}

        df = self.metrics_df.copy()
        best_name = str(df["RMSE"].astype(float).idxmin())
        best = df.loc[best_name]
        return {
            "rmse": float(best["RMSE"]),
            "mae": float(best["MAE"]),
            "best_model": best_name,
            "all_models": [
                {"model": str(idx), "rmse": float(row["RMSE"]), "mae": float(row["MAE"])}
                for idx, row in df.iterrows()
            ],
        }


service = RecommenderService()

app = FastAPI(title="Movie Recommendation Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "ready": service.ready}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return service.metrics_payload()


@app.get("/recommend")
def recommend(
    user_id: int = Query(..., description="Raw MovieLens user ID"),
    k: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    try:
        recs = service.recommend(user_id=user_id, k=k)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    return {"user_id": user_id, "recommendations": recs}
