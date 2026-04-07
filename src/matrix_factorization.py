"""
matrix_factorization.py
-----------------------
Matrix Factorization with biases:

  r_hat_ui = mu + b_u + b_i + p_u^T q_i

  1. MatrixFactorization  - pure NumPy SGD (from scratch, for understanding)
  2. SurpriseMF           - wrapper around surprise.SVD (fast, for experiments)

MF sensitivity fix:
  - d sweep uses 50% subsample (dense enough for larger d to win)
  - 30 epochs in sweep so all d values converge properly
  - lambda sweep uses best_d from d sweep
"""

import numpy as np
import pandas as pd
import time

try:
    from surprise import Dataset, Reader, SVD, accuracy
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


# ─────────────────────────────────────────────────────────────────────
# 1. From-scratch NumPy SGD MF (educational)
# ─────────────────────────────────────────────────────────────────────

class MatrixFactorization:
    """
    Explicit-feedback MF with user/item biases trained by SGD.
    Use SurpriseMF for actual experiments — this is for report/understanding.
    """

    def __init__(
        self,
        n_users:  int,
        n_items:  int,
        d:        int   = 50,
        lam:      float = 0.02,
        lr:       float = 0.005,
        n_epochs: int   = 20,
    ):
        self.n_users  = n_users
        self.n_items  = n_items
        self.d        = d
        self.lam      = lam
        self.lr       = lr
        self.n_epochs = n_epochs
        self.mu       = 0.0
        self.b_u      = np.zeros(n_users)
        self.b_i      = np.zeros(n_items)
        self.P        = np.random.normal(0, 0.1, (n_users, d))
        self.Q        = np.random.normal(0, 0.1, (n_items, d))
        self.train_rmse_history = []

    def fit(self, train: pd.DataFrame):
        self.mu = train["rating"].mean()
        users   = train["user_idx"].values.astype(int)
        items   = train["movie_idx"].values.astype(int)
        ratings = train["rating"].values.astype(float)
        n       = len(ratings)

        for epoch in range(self.n_epochs):
            idx  = np.random.permutation(n)
            u_sh = users[idx]
            i_sh = items[idx]
            r_sh = ratings[idx]

            for pos in range(n):
                u   = u_sh[pos]; i = i_sh[pos]; r = r_sh[pos]
                err = r - (self.mu + self.b_u[u] + self.b_i[i]
                           + self.P[u] @ self.Q[i])
                self.b_u[u] += self.lr * (err - self.lam * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.lam * self.b_i[i])
                p_u          = self.P[u].copy()
                self.P[u]   += self.lr * (err * self.Q[i] - self.lam * self.P[u])
                self.Q[i]   += self.lr * (err * p_u       - self.lam * self.Q[i])

            preds      = (self.mu + self.b_u[users] + self.b_i[items]
                          + np.sum(self.P[users] * self.Q[items], axis=1))
            train_rmse = float(np.sqrt(np.mean((ratings - preds) ** 2)))
            self.train_rmse_history.append(train_rmse)
            if (epoch + 1) % 5 == 0:
                print(f"  [MF-scratch] epoch {epoch+1:>3}/{self.n_epochs}  "
                      f"train RMSE={train_rmse:.4f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        users = df["user_idx"].values.astype(int)
        items = df["movie_idx"].values.astype(int)
        preds = (self.mu + self.b_u[users] + self.b_i[items]
                 + np.sum(self.P[users] * self.Q[items], axis=1))
        return np.clip(preds, 0.5, 5.0)

    def evaluate(self, test: pd.DataFrame, label="MF-scratch"):
        return rating_metrics(test["rating"].values, self.predict(test), label)


# ─────────────────────────────────────────────────────────────────────
# 2. SurpriseMF wrapper (fast — use for all experiments)
# ─────────────────────────────────────────────────────────────────────

class SurpriseMF:
    """
    Wrapper around surprise.SVD (biased MF with SGD).

    Parameters
    ----------
    d        : number of latent factors
    n_epochs : SGD epochs
    lr_all   : learning rate
    reg_all  : regularisation weight
    """

    def __init__(
        self,
        d:        int   = 50,
        n_epochs: int   = 20,
        lr_all:   float = 0.005,
        reg_all:  float = 0.02,
    ):
        if not SURPRISE_OK:
            raise RuntimeError(
                "scikit-surprise required. Run: python -m pip install scikit-surprise"
            )
        self.d        = d
        self.n_epochs = n_epochs
        self.lr_all   = lr_all
        self.reg_all  = reg_all
        self.algo     = None

    def fit(self, train: pd.DataFrame):
        t0    = time.time()
        train = _ensure_str_ids(train)
        data  = _build_surprise_dataset(train)
        trainset = data.build_full_trainset()
        self.algo = SVD(
            n_factors = self.d,
            n_epochs  = self.n_epochs,
            lr_all    = self.lr_all,
            reg_all   = self.reg_all,
            biased    = True,
            verbose   = False,
        )
        self.algo.fit(trainset)
        print(f"[SurpriseMF] fitted  d={self.d}  epochs={self.n_epochs}  "
              f"reg={self.reg_all}  ({time.time()-t0:.1f}s)")
        return self

    def predict_df(self, test: pd.DataFrame) -> np.ndarray:
        test    = _ensure_str_ids(test)
        testset = _to_surprise_testset(test)
        preds   = self.algo.test(testset)
        return np.array([p.est for p in preds])

    def evaluate(self, test: pd.DataFrame, label="MF"):
        return rating_metrics(test["rating"].values, self.predict_df(test), label)


# ─────────────────────────────────────────────────────────────────────
# Sensitivity: vary d
# Uses 50% subsample so larger d has enough data to win over d=10
# Uses 30 epochs so all models fully converge
# ─────────────────────────────────────────────────────────────────────

def mf_sensitivity(
    train:       pd.DataFrame,
    test:        pd.DataFrame,
    d_values=None,
    sample_frac: float = 0.5,
    seed:        int   = 42,
) -> pd.DataFrame:
    """
    Sweep over latent dimensions.
    Uses 50% subsample + 30 epochs for reliable results.
    Returns DataFrame indexed by d.
    """
    if d_values is None:
        d_values = [10, 20, 50, 100, 150, 200]

    train_s = train.sample(frac=sample_frac, random_state=seed).copy()
    test_s  = test.sample(frac=0.3,          random_state=seed).copy()
    train_s = _ensure_str_ids(train_s)
    test_s  = _ensure_str_ids(test_s)

    print(f"[mf_sensitivity] {len(train_s):,} train / {len(test_s):,} test rows")

    rows = []
    for d in d_values:
        model   = SurpriseMF(d=d, n_epochs=30, lr_all=0.005, reg_all=0.02)
        model.fit(train_s)
        preds   = model.predict_df(test_s)
        metrics = rating_metrics(test_s["rating"].values, preds, label=f"MF d={d}")
        metrics["d"] = d
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("d")


# ─────────────────────────────────────────────────────────────────────
# Sensitivity: vary lambda  (uses best_d from d sweep)
# ─────────────────────────────────────────────────────────────────────

def mf_lambda_sensitivity(
    train:         pd.DataFrame,
    test:          pd.DataFrame,
    lambda_values=None,
    best_d:        int   = 50,
    sample_frac:   float = 0.5,
    seed:          int   = 42,
) -> pd.DataFrame:
    """
    Sweep over regularisation weights using the best d from mf_sensitivity.
    Returns DataFrame indexed by lambda.
    """
    if lambda_values is None:
        lambda_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    train_s = train.sample(frac=sample_frac, random_state=seed).copy()
    test_s  = test.sample(frac=0.3,          random_state=seed).copy()
    train_s = _ensure_str_ids(train_s)
    test_s  = _ensure_str_ids(test_s)

    print(f"[mf_lambda_sensitivity] best_d={best_d}  "
          f"{len(train_s):,} train / {len(test_s):,} test rows")

    rows = []
    for lam in lambda_values:
        model   = SurpriseMF(d=best_d, n_epochs=30, reg_all=lam)
        model.fit(train_s)
        preds   = model.predict_df(test_s)
        metrics = rating_metrics(test_s["rating"].values, preds, label=f"MF lam={lam}")
        metrics["lambda"] = lam
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("lambda")


# ─────────────────────────────────────────────────────────────────────
# Top-K recommendations
# ─────────────────────────────────────────────────────────────────────

def get_topk_recs_mf(
    model:               SurpriseMF,
    str_user_id:         str,
    all_str_movie_ids,
    rated_str_movie_ids,
    k: int = 10,
) -> list:
    unrated = [m for m in all_str_movie_ids if m not in rated_str_movie_ids]
    scores  = [(m, model.algo.predict(str_user_id, m).est) for m in unrated]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]