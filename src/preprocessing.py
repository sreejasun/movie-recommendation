"""
preprocessing.py
----------------
Data loading, cleaning, and per-user train/test splitting for
the MovieLens 20M dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_ratings(data_dir: str, sample_frac: float = None, seed: int = 42) -> pd.DataFrame:
    path = os.path.join(data_dir, "rating.csv")
    print(f"[load_ratings] reading {path} ...")
    df = pd.read_csv(path)

    df = df.dropna(subset=["userId", "movieId", "rating"])
    df = df.drop_duplicates(subset=["userId", "movieId"])
    df["userId"]  = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"]  = df["rating"].astype(float)

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        print(f"[load_ratings] sampled {len(df):,} rows (frac={sample_frac})")
    else:
        print(f"[load_ratings] loaded {len(df):,} rows")

    return df


def load_movies(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "movie.csv")
    return pd.read_csv(path)


def encode_ids(df: pd.DataFrame):
    """
    Map userId / movieId to contiguous zero-based integer indices.
    Also adds str_userId and str_movieId columns for Surprise compatibility.
    """
    user_enc  = LabelEncoder()
    movie_enc = LabelEncoder()

    df = df.copy()
    df["user_idx"]    = user_enc.fit_transform(df["userId"])
    df["movie_idx"]   = movie_enc.fit_transform(df["movieId"])

    # String versions needed by Surprise for consistent ID matching
    df["str_userId"]  = df["userId"].astype(str)
    df["str_movieId"] = df["movieId"].astype(str)

    n_users  = df["user_idx"].nunique()
    n_movies = df["movie_idx"].nunique()
    print(f"[encode_ids] {n_users:,} users  |  {n_movies:,} movies")
    return df, user_enc, movie_enc


def per_user_split(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42):
    """
    Hold out the most recent test_ratio fraction of each user's ratings
    (sorted by timestamp) as the test set.
    """
    np.random.seed(seed)
    train_parts, test_parts = [], []

    for _, group in df.groupby("user_idx"):
        group  = group.sort_values("timestamp")
        n      = len(group)
        n_test = max(1, int(n * test_ratio)) if n > 1 else 0

        if n_test == 0:
            train_parts.append(group)
        else:
            train_parts.append(group.iloc[:-n_test])
            test_parts.append(group.iloc[-n_test:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test  = pd.concat(test_parts).reset_index(drop=True)
    print(f"[per_user_split] train={len(train):,}  test={len(test):,}")
    return train, test


def sparsity_report(df: pd.DataFrame) -> None:
    n_users  = df["user_idx"].nunique()
    n_movies = df["movie_idx"].nunique()
    density  = len(df) / (n_users * n_movies)
    print(f"\n{'─'*40}")
    print(f"  Users   : {n_users:>10,}")
    print(f"  Movies  : {n_movies:>10,}")
    print(f"  Ratings : {len(df):>10,}")
    print(f"  Density : {density:.4%}")
    print(f"{'─'*40}\n")