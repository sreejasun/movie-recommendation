from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.matrix_factorization import SurpriseMF, get_topk_recs_mf, load_mf_model


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MF_MODEL_PATH = RESULTS_DIR / "mf_model.pkl"


class RecommenderService:
    """
    Serves Top-K recommendations using Matrix Factorization (Surprise SVD) when available.
    Falls back to a bias model (μ + b_u + b_i) if MF cannot be trained or loaded.
    """

    def __init__(self) -> None:
        self.mu: float = 0.0
        self.user_bias: dict[int, float] = {}
        self.item_bias: dict[int, float] = {}
        self.movies: pd.DataFrame = pd.DataFrame()
        self.all_movie_ids: np.ndarray = np.array([], dtype=int)
        self.all_str_movie_ids: list[str] = []
        self.user_rated: dict[int, set[int]] = {}
        self.metrics_df: pd.DataFrame = pd.DataFrame()
        self.mf_model: SurpriseMF | None = None
        self.mf_params: dict[str, Any] = {}
        self.engine: str = "bias"  # "mf" | "bias"
        self.ready: bool = False
        self._load()

    def _mf_config(self) -> dict[str, Any]:
        return {
            "d": int(os.environ.get("MF_D", "10")),
            "n_epochs": int(os.environ.get("MF_EPOCHS", "20")),
            "lr_all": float(os.environ.get("MF_LR", "0.005")),
            "reg_all": float(os.environ.get("MF_REG", "0.02")),
        }

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

        ratings = pd.read_csv(rating_path, usecols=["userId", "movieId", "rating"])
        ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        return ratings

    def _load_metrics(self) -> pd.DataFrame:
        metrics_path = RESULTS_DIR / "all_model_results.csv"
        if metrics_path.exists():
            return pd.read_csv(metrics_path, index_col=0)
        return pd.DataFrame(columns=["RMSE", "MAE"])

    def _ensure_str_columns(self, train: pd.DataFrame) -> pd.DataFrame:
        df = train.copy()
        if "str_userId" not in df.columns:
            df["str_userId"] = df["userId"].astype(str)
        if "str_movieId" not in df.columns:
            df["str_movieId"] = df["movieId"].astype(str)
        return df

    def _try_load_mf_pickle(self) -> None:
        """Load pretrained MF from disk. Training is done offline (see backend/train_mf_server.py)."""
        cfg = self._mf_config()
        self.mf_params = {
            "name": "matrix_factorization",
            "implementation": "Surprise SVD (biased MF)",
            "latent_factors": cfg["d"],
            "epochs": cfg["n_epochs"],
            "learning_rate": cfg["lr_all"],
            "regularization": cfg["reg_all"],
        }

        if not MF_MODEL_PATH.exists():
            print(
                f"[RecommenderService] No {MF_MODEL_PATH.name} — "
                "recommendations use bias fallback. Run: python backend/train_mf_server.py"
            )
            self.mf_model = None
            self.engine = "bias"
            return

        try:
            algo = load_mf_model(MF_MODEL_PATH)
            mf = SurpriseMF(
                d=cfg["d"],
                n_epochs=cfg["n_epochs"],
                lr_all=cfg["lr_all"],
                reg_all=cfg["reg_all"],
            )
            mf.algo = algo
            self.mf_model = mf
            self.engine = "mf"
            print(f"[RecommenderService] MF loaded from {MF_MODEL_PATH}")
        except Exception as e:
            print(f"[RecommenderService] MF load failed ({e}); using bias model.")
            self.mf_model = None
            self.engine = "bias"

    def _load(self) -> None:
        train = self._load_train()
        movies_path = DATA_DIR / "movie.csv"
        if not movies_path.exists():
            raise FileNotFoundError("Missing data/movie.csv")

        self.movies = pd.read_csv(movies_path, usecols=["movieId", "title"]).copy()
        self.movies["movieId"] = self.movies["movieId"].astype(int)
        self.movies = self.movies.drop_duplicates(subset=["movieId"])
        self.all_movie_ids = self.movies["movieId"].to_numpy(dtype=int)
        self.all_str_movie_ids = [str(int(m)) for m in self.all_movie_ids]

        self.mu = float(train["rating"].mean())
        user_mean = train.groupby("userId")["rating"].mean()
        item_mean = train.groupby("movieId")["rating"].mean()
        self.user_bias = (user_mean - self.mu).to_dict()
        self.item_bias = (item_mean - self.mu).to_dict()

        rated_pairs = train.groupby("userId")["movieId"].apply(lambda s: set(s.astype(int)))
        self.user_rated = rated_pairs.to_dict()
        self.metrics_df = self._load_metrics()

        self._try_load_mf_pickle()
        self.ready = True

    def _predict_bias(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        bu = self.user_bias.get(user_id, 0.0)
        bi = np.array([self.item_bias.get(int(mid), 0.0) for mid in movie_ids], dtype=float)
        pred = self.mu + bu + bi
        return np.clip(pred, 0.5, 5.0)

    def _recommend_bias(self, user_id: int, k: int) -> list[dict[str, Any]]:
        seen = self.user_rated[user_id]
        candidates = np.array([mid for mid in self.all_movie_ids if int(mid) not in seen], dtype=int)
        if len(candidates) == 0:
            return []
        scores = self._predict_bias(user_id, candidates)
        # Primary: higher score; secondary: lower movie id (stable tie-break)
        order = np.lexsort((candidates.astype(np.int64), -scores))
        top_idx = order[:k]
        top_ids = candidates[top_idx]
        top_scores = scores[top_idx]
        title_map = dict(zip(self.movies["movieId"].astype(int), self.movies["title"]))
        return [
            {
                "movie_id": int(mid),
                "title": title_map.get(int(mid), str(mid)),
                "predicted_rating": round(float(sc), 3),
            }
            for mid, sc in zip(top_ids, top_scores)
        ]

    def _recommend_mf(self, user_id: int, k: int) -> list[dict[str, Any]]:
        assert self.mf_model is not None
        seen = self.user_rated[user_id]
        rated_str = {str(int(x)) for x in seen}
        top = get_topk_recs_mf(
            self.mf_model,
            str(int(user_id)),
            self.all_str_movie_ids,
            rated_str,
            k=k,
        )
        title_map = dict(zip(self.movies["movieId"].astype(int), self.movies["title"]))
        out = []
        for mid_str, score in top:
            mid_i = int(mid_str)
            out.append(
                {
                    "movie_id": mid_i,
                    "title": title_map.get(mid_i, str(mid_i)),
                    "predicted_rating": round(float(score), 3),
                }
            )
        return out

    def recommend(self, user_id: int, k: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if user_id not in self.user_rated:
            raise KeyError(f"user_id={user_id} not found")

        if self.engine == "mf" and self.mf_model is not None:
            recs = self._recommend_mf(user_id, k)
            meta = {**self.mf_params, "engine": "mf"}
            return recs, meta

        recs = self._recommend_bias(user_id, k)
        meta = {
            "name": "bias_model",
            "implementation": "μ + b_user + b_item (training-set biases)",
            "engine": "bias",
        }
        return recs, meta

    def metrics_payload(self) -> dict[str, Any]:
        if self.metrics_df.empty:
            base: dict[str, Any] = {
                "rmse": None,
                "mae": None,
                "best_model": None,
                "all_models": [],
            }
        else:
            df = self.metrics_df.copy()
            best_name = str(df["RMSE"].astype(float).idxmin())
            best = df.loc[best_name]
            base = {
                "rmse": float(best["RMSE"]),
                "mae": float(best["MAE"]),
                "best_model": best_name,
                "all_models": [
                    {"model": str(idx), "rmse": float(row["RMSE"]), "mae": float(row["MAE"])}
                    for idx, row in df.iterrows()
                ],
            }
        base["recommendation_engine"] = self.engine
        base["serving_model"] = self.mf_params if self.engine == "mf" else {"name": "bias_model"}
        return base


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
    return {
        "ok": True,
        "ready": service.ready,
        "recommendation_engine": service.engine,
        "serving_model": service.mf_params if service.engine == "mf" else {"name": "bias_model"},
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return service.metrics_payload()


@app.get("/recommend")
def recommend(
    user_id: int = Query(..., description="Raw MovieLens user ID"),
    k: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    try:
        recs, model_meta = service.recommend(user_id=user_id, k=k)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    return {
        "user_id": user_id,
        "model": model_meta,
        "recommendations": recs,
    }
