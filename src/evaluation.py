"""
evaluation.py
-------------
Shared evaluation metrics used by every model:
  - RMSE, MAE  (rating prediction)
  - Precision@K, Recall@K, NDCG@K  (Top-K recommendation)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error


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


# ─────────────────────────────────────────────
# Confusion matrices (classification view of ratings)
# ─────────────────────────────────────────────


def binary_like_confusion_matrix(
    y_true,
    y_pred,
    threshold: float = 4.0,
) -> np.ndarray:
    """
    Treat "like" as rating >= threshold (same spirit as Top-K relevance).

    Rows = true class, Cols = predicted class.
    Order: 0 = dislike (< threshold), 1 = like (>= threshold).

    Returns
    -------
    cm : (2, 2) int array
         [[TN, FP], [FN, TP]] in terms of dislike/like with labels [0, 1].
    """
    yt = (np.asarray(y_true, dtype=float) >= threshold).astype(int)
    yp = (np.asarray(y_pred, dtype=float) >= threshold).astype(int)
    return confusion_matrix(yt, yp, labels=[0, 1])


def binary_confusion_labels(threshold: float) -> tuple[list[str], list[str]]:
    lo = f"Dislike\n(< {threshold:g})"
    hi = f"Like\n(≥ {threshold:g})"
    return [lo, hi], [lo, hi]


def half_star_bucket(y: np.ndarray) -> np.ndarray:
    """Nearest half-star in [0.5, 5.0] (MovieLens scale)."""
    y = np.asarray(y, dtype=float)
    return np.clip(np.round(y * 2.0) / 2.0, 0.5, 5.0)


def _half_star_index(r: float) -> int:
    """Map a rating to class index 0..9 for half-stars 0.5 .. 5.0."""
    r = float(np.clip(np.round(float(r) * 2.0) / 2.0, 0.5, 5.0))
    return int(round((r - 0.5) / 0.5))


def half_star_confusion_matrix(y_true, y_pred) -> tuple[np.ndarray, list[float]]:
    """
    Confusion matrix over 10 half-star classes (0.5 .. 5.0).

    Returns
    -------
    cm : (10, 10) int
    labels : list of rating values in row/column order
    """
    labels = [0.5 + 0.5 * i for i in range(10)]
    yt = half_star_bucket(y_true)
    yp = half_star_bucket(y_pred)
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(yt, yp):
        ti = _half_star_index(t)
        pi = _half_star_index(p)
        cm[ti, pi] += 1
    return cm, labels


def confusion_matrix_to_csv(
    cm: np.ndarray,
    row_labels: list,
    col_labels: list,
    path: str,
) -> None:
    pd.DataFrame(cm, index=row_labels, columns=col_labels).to_csv(path)


def save_binary_confusion_figure(
    cm: np.ndarray,
    title: str,
    png_path: str,
    threshold: float = 4.0,
) -> None:
    import matplotlib.pyplot as plt

    row_labels, col_labels = binary_confusion_labels(threshold)
    row_csv = [s.replace("\n", " ") for s in row_labels]
    col_csv = [s.replace("\n", " ") for s in col_labels]
    confusion_matrix_to_csv(cm, row_csv, col_csv, png_path.replace(".png", ".csv"))

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=col_labels,
        yticklabels=row_labels,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_halfstar_confusion_figure(
    cm: np.ndarray,
    labels: list[float],
    title: str,
    png_path: str,
) -> None:
    import matplotlib.pyplot as plt

    ls = [str(x) for x in labels]
    confusion_matrix_to_csv(cm, ls, ls, png_path.replace(".png", ".csv"))

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=ls,
        yticklabels=ls,
        ylabel="True rating (half-stars)",
        xlabel="Predicted rating (half-stars)",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_all_binary_confusion_grid(
    named_matrices: list[tuple[str, np.ndarray]],
    png_path: str,
    threshold: float = 4.0,
) -> None:
    """One figure with a heatmap per model (same colour scale for comparability)."""
    import matplotlib.pyplot as plt

    n = len(named_matrices)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    row_labels, col_labels = binary_confusion_labels(threshold)
    vmax = max(m.max() for _, m in named_matrices) if named_matrices else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, cm) in zip(axes, named_matrices):
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=11)
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels(col_labels, fontsize=7)
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        thresh = vmax / 2.0 if vmax > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if cm[i, j] > thresh else "black",
                )
    for ax in axes[len(named_matrices) :]:
        ax.axis("off")
    used_axes = list(axes[: len(named_matrices)])
    fig.suptitle(
        f"Binary like / dislike confusion (threshold = {threshold:g} stars)",
        fontsize=13,
        y=1.02,
    )
    fig.colorbar(im, ax=used_axes, fraction=0.02, pad=0.04, label="Count")
    fig.subplots_adjust(top=0.88, wspace=0.35, hspace=0.35)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def binary_rates_from_confusion(cm: np.ndarray) -> dict[str, float]:
    """TN, FP, FN, TP and accuracy from a 2x2 like/dislike matrix."""
    tn, fp, fn, tp = cm.ravel().astype(float)
    tot = tn + fp + fn + tp
    acc = (tp + tn) / tot if tot > 0 else 0.0
    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": acc,
        "precision_like": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall_like": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    }
