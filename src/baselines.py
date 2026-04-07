"""
baselines.py
------------
Four baseline rating predictors:
  1. GlobalMean   – predict the training-set mean for every pair
  2. UserMean     – predict each user's mean rating
  3. ItemMean     – predict each item's mean rating
  4. BiasModel    – r_hat = μ + b_u + b_i  (SGD, regularised)
"""

import numpy as np
import pandas as pd
from src.evaluation import rating_metrics


# ─────────────────────────────────────────────
# 1. Global Mean
# ─────────────────────────────────────────────

class GlobalMean:
    def __init__(self):
        self.mu = None

    def fit(self, train: pd.DataFrame):
        self.mu = train["rating"].mean()
        print(f"[GlobalMean] μ = {self.mu:.4f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mu)

    def evaluate(self, test: pd.DataFrame, label="GlobalMean"):
        preds = self.predict(test)
        return rating_metrics(test["rating"].values, preds, label)


# ─────────────────────────────────────────────
# 2. User Mean
# ─────────────────────────────────────────────

class UserMean:
    def __init__(self):
        self.mu         = None
        self.user_means = {}

    def fit(self, train: pd.DataFrame):
        self.mu         = train["rating"].mean()
        self.user_means = train.groupby("user_idx")["rating"].mean().to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["user_idx"].map(self.user_means).fillna(self.mu).values

    def evaluate(self, test: pd.DataFrame, label="UserMean"):
        preds = self.predict(test)
        return rating_metrics(test["rating"].values, preds, label)


# ─────────────────────────────────────────────
# 3. Item Mean
# ─────────────────────────────────────────────

class ItemMean:
    def __init__(self):
        self.mu         = None
        self.item_means = {}

    def fit(self, train: pd.DataFrame):
        self.mu         = train["rating"].mean()
        self.item_means = train.groupby("movie_idx")["rating"].mean().to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["movie_idx"].map(self.item_means).fillna(self.mu).values

    def evaluate(self, test: pd.DataFrame, label="ItemMean"):
        preds = self.predict(test)
        return rating_metrics(test["rating"].values, preds, label)


# ─────────────────────────────────────────────
# 4. Bias Model  (r_hat = μ + b_u + b_i)
# ─────────────────────────────────────────────

class BiasModel:
    """
    Learns user and item biases by minimising regularised squared error
    with Stochastic Gradient Descent.

        L = Σ (r_ui − μ − b_u − b_i)² + λ (b_u² + b_i²)

    Parameters
    ----------
    n_users, n_items : dimensions of the rating matrix
    lam              : L2 regularisation weight
    lr               : SGD learning rate
    n_epochs         : number of full passes over training data
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        lam: float = 0.1,
        lr: float  = 0.005,
        n_epochs: int = 20,
    ):
        self.n_users  = n_users
        self.n_items  = n_items
        self.lam      = lam
        self.lr       = lr
        self.n_epochs = n_epochs
        self.mu       = 0.0
        self.b_u      = np.zeros(n_users)
        self.b_i      = np.zeros(n_items)

    def fit(self, train: pd.DataFrame):
        self.mu  = train["rating"].mean()
        users    = train["user_idx"].values
        items    = train["movie_idx"].values
        ratings  = train["rating"].values

        for epoch in range(self.n_epochs):
            # shuffle each epoch
            idx = np.random.permutation(len(ratings))
            u_arr, i_arr, r_arr = users[idx], items[idx], ratings[idx]

            for u, i, r in zip(u_arr, i_arr, r_arr):
                err         = r - (self.mu + self.b_u[u] + self.b_i[i])
                self.b_u[u] += self.lr * (err - self.lam * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.lam * self.b_i[i])

            if (epoch + 1) % 5 == 0:
                preds = self.mu + self.b_u[users] + self.b_i[items]
                rmse  = np.sqrt(np.mean((ratings - preds) ** 2))
                print(f"  [BiasModel] epoch {epoch+1:>3}/{self.n_epochs}  train RMSE={rmse:.4f}")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        users = df["user_idx"].values
        items = df["movie_idx"].values
        preds = self.mu + self.b_u[users] + self.b_i[items]
        return np.clip(preds, 0.5, 5.0)

    def evaluate(self, test: pd.DataFrame, label="BiasModel"):
        preds = self.predict(test)
        return rating_metrics(test["rating"].values, preds, label)
