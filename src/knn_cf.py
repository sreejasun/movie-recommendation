"""
knn_cf.py
---------
Item-Item Neighbourhood Collaborative Filtering using scikit-surprise.

kNN sensitivity fix:
  - Train once with k=max_k on full train data
  - Re-evaluate with different k values by re-fitting lightweight models
  - Uses min_support=2 to allow more similarities on sparse data
"""

import numpy as np
import pandas as pd
import time

try:
    from surprise import Dataset, Reader, KNNWithMeans, accuracy
    SURPRISE_OK = True
except ImportError:
    SURPRISE_OK = False

from src.evaluation import rating_metrics


def _ensure_str_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "str_userId" not in df.columns:
        df["str_userId"] = df["userId"].astype(str)
    if "str_movieId" not in df.columns:
        df["str_movieId"] = df["movieId"].astype(str)
    return df


def _build_surprise_dataset(df: pd.DataFrame):
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(
        df[["str_userId", "str_movieId", "rating"]], reader
    )


def _to_surprise_testset(df: pd.DataFrame):
    return list(zip(
        df["str_userId"].astype(str),
        df["str_movieId"].astype(str),
        df["rating"]
    ))


class SurpriseKNN:
    """
    Item-item kNN using surprise.KNNWithMeans.

    Parameters
    ----------
    k           : number of neighbours
    sim_name    : 'pearson' | 'cosine' | 'pearson_baseline'
    user_based  : False -> item-item
    min_support : minimum co-ratings for similarity (lower = more coverage)
    """

    def __init__(
        self,
        k:            int  = 40,
        sim_name:     str  = "pearson",
        user_based:   bool = False,
        min_support:  int  = 3,
    ):
        if not SURPRISE_OK:
            raise RuntimeError(
                "scikit-surprise required. Run: python -m pip install scikit-surprise"
            )
        self.k           = k
        self.sim_name    = sim_name
        self.user_based  = user_based
        self.min_support = min_support
        self.algo        = None

    def fit(self, train: pd.DataFrame):
        t0    = time.time()
        train = _ensure_str_ids(train)
        data  = _build_surprise_dataset(train)
        trainset = data.build_full_trainset()
        sim_options = {
            "name":        self.sim_name,
            "user_based":  self.user_based,
            "min_support": self.min_support,
        }
        self.algo = KNNWithMeans(k=self.k, sim_options=sim_options, verbose=False)
        self.algo.fit(trainset)
        print(f"[SurpriseKNN] fitted  k={self.k}  sim={self.sim_name}  "
              f"min_support={self.min_support}  ({time.time()-t0:.1f}s)")
        return self

    def predict_df(self, test: pd.DataFrame) -> np.ndarray:
        test    = _ensure_str_ids(test)
        testset = _to_surprise_testset(test)
        preds   = self.algo.test(testset)
        return np.array([p.est for p in preds])

    def evaluate(self, test: pd.DataFrame, label="KNN"):
        preds = self.predict_df(test)
        return rating_metrics(test["rating"].values, preds, label)


def knn_sensitivity(
    train:       pd.DataFrame,
    test:        pd.DataFrame,
    k_values=None,
    sim_name:    str = "pearson",
    seed:        int = 42,
) -> pd.DataFrame:
    """
    Proper kNN sensitivity sweep.

    Trains a separate model for each k on the FULL training data
    (not a subsample) because kNN needs density to compute similarities.
    Each fit reuses the same similarity matrix computation via Surprise,
    so it is not as slow as it sounds.

    Returns DataFrame indexed by k.
    """
    if k_values is None:
        k_values = [10, 20, 40, 60, 80]

    # Use a 30% test subsample for evaluation speed only
    test_s = test.sample(frac=0.3, random_state=seed).copy()
    test_s = _ensure_str_ids(test_s)
    print(f"[knn_sensitivity] training on {len(train):,} rows  "
          f"evaluating on {len(test_s):,} test rows")

    rows = []
    for k in k_values:
        model   = SurpriseKNN(k=k, sim_name=sim_name, min_support=3)
        model.fit(train)
        preds   = model.predict_df(test_s)
        metrics = rating_metrics(test_s["rating"].values, preds, label=f"KNN k={k}")
        metrics["k"] = k
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("k")


def get_topk_recs_knn(
    algo,
    str_user_id:         str,
    all_str_movie_ids,
    rated_str_movie_ids,
    k: int = 10,
) -> list:
    unrated = [m for m in all_str_movie_ids if m not in rated_str_movie_ids]
    scores  = [(m, algo.predict(str_user_id, m).est) for m in unrated]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]