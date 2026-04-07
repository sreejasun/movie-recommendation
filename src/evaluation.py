"""
evaluation.py
-------------
Shared evaluation metrics used by every model:
  - RMSE, MAE  (rating prediction)
  - Precision@K, Recall@K, NDCG@K  (Top-K recommendation)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────
# Rating-prediction metrics
# ─────────────────────────────────────────────

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rating_metrics(y_true, y_pred, label: str = "") -> dict:
    r = rmse(y_true, y_pred)
    m = mae(y_true, y_pred)
    if label:
        print(f"[{label}]  RMSE={r:.4f}  MAE={m:.4f}")
    return {"RMSE": r, "MAE": m}


# ─────────────────────────────────────────────
# Top-K recommendation metrics
# ─────────────────────────────────────────────

def dcg_at_k(relevances, k):
    """Discounted Cumulative Gain at K."""
    relevances = np.asarray(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(relevances, k):
    """Normalised DCG at K."""
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal_dcg


def topk_metrics(
    predictions_df: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0,
) -> dict:
    """
    Compute Precision@K, Recall@K, and NDCG@K.

    Parameters
    ----------
    predictions_df : DataFrame with columns
                     [user_idx, movie_idx, true_rating, pred_rating]
    k              : cut-off rank
    threshold      : minimum true rating to count as relevant

    Returns
    -------
    dict with keys Precision@K, Recall@K, NDCG@K
    """
    precisions, recalls, ndcgs = [], [], []

    for _, group in predictions_df.groupby("user_idx"):
        group       = group.sort_values("pred_rating", ascending=False)
        top_k       = group.head(k)
        rel_in_topk = (top_k["true_rating"] >= threshold).astype(int).tolist()
        rel_total   = (group["true_rating"] >= threshold).sum()

        precisions.append(sum(rel_in_topk) / k)
        recalls.append(sum(rel_in_topk) / rel_total if rel_total > 0 else 0.0)
        ndcgs.append(ndcg_at_k(rel_in_topk, k))

    result = {
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}":    float(np.mean(recalls)),
        f"NDCG@{k}":      float(np.mean(ndcgs)),
    }
    for key, val in result.items():
        print(f"  {key}: {val:.4f}")
    return result
