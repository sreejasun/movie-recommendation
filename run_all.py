"""
run_all.py
----------
End-to-end pipeline: preprocessing -> baselines -> kNN -> MF -> results.

Usage
-----
  python run_all.py --data-dir data/ --sample 0.1   # quick dev run
  python run_all.py --data-dir data/                 # full 20M run
  python run_all.py --data-dir data/ --no-knn        # skip kNN
"""

import argparse
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing        import load_ratings, load_movies, encode_ids, per_user_split, sparsity_report
from src.evaluation           import (
    rating_metrics,
    topk_metrics,
    binary_like_confusion_matrix,
    binary_rates_from_confusion,
    half_star_confusion_matrix,
    save_all_binary_confusion_grid,
    save_binary_confusion_figure,
    save_halfstar_confusion_figure,
)
from src.baselines            import GlobalMean, UserMean, ItemMean, BiasModel
from src.knn_cf               import SurpriseKNN, knn_sensitivity, get_topk_recs_knn
from src.matrix_factorization import (SurpriseMF, mf_sensitivity,
                                      mf_lambda_sensitivity, get_topk_recs_mf)

os.makedirs("results", exist_ok=True)
SEED = 42
np.random.seed(SEED)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/")
    p.add_argument("--sample",   type=float, default=None)
    p.add_argument("--no-knn",   action="store_true")
    return p.parse_args()


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def _safe_filename_fragment(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def save_sensitivity_plot(df, title, xlabel, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["RMSE"].plot(ax=axes[0], marker="o", color="steelblue")
    axes[0].set_title(f"{title} - RMSE")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("RMSE")

    df["MAE"].plot(ax=axes[1], marker="s", color="coral")
    axes[1].set_title(f"{title} - MAE")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("MAE")

    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: results/{filename}")


def main():
    args    = parse_args()
    t_start = time.time()

    # ── 1. Load & preprocess ──────────────────────────────────────────
    section("STEP 1: Data loading & preprocessing")

    ratings = load_ratings(args.data_dir, sample_frac=args.sample, seed=SEED)
    movies  = load_movies(args.data_dir)
    ratings, user_enc, movie_enc = encode_ids(ratings)
    sparsity_report(ratings)

    train, test = per_user_split(ratings, test_ratio=0.2, seed=SEED)

    n_users  = int(ratings["user_idx"].max()) + 1
    n_movies = int(ratings["movie_idx"].max()) + 1

    train.to_parquet("data/train.parquet", index=False)
    test.to_parquet("data/test.parquet",   index=False)
    print(f"Splits saved.  n_users={n_users:,}  n_movies={n_movies:,}")

    results = {}

    # ── 2. Baselines ──────────────────────────────────────────────────
    section("STEP 2: Baseline models")

    gm = GlobalMean().fit(train)
    results["GlobalMean"] = gm.evaluate(test)

    um = UserMean().fit(train)
    results["UserMean"] = um.evaluate(test)

    im = ItemMean().fit(train)
    results["ItemMean"] = im.evaluate(test)

    print("Training BiasModel ...")
    bm = BiasModel(n_users=n_users, n_items=n_movies, lam=0.1, lr=0.005, n_epochs=20)
    bm.fit(train)
    results["BiasModel"] = bm.evaluate(test)

    # ── 3. kNN ────────────────────────────────────────────────────────
    knn = None
    if not args.no_knn:
        section("STEP 3: kNN Collaborative Filtering")
        t0 = time.time()

        # Sensitivity sweep — trains on full data, evaluates on 30% test subsample
        print("Running k sensitivity sweep ...")
        df_knn_sens = knn_sensitivity(
            train, test,
            k_values=[10, 20, 40, 60, 80],
            seed=SEED
        )
        df_knn_sens.to_csv("results/knn_sensitivity.csv")
        print(df_knn_sens)
        save_sensitivity_plot(df_knn_sens,
                              "kNN Neighbourhood Size k",
                              "k (number of neighbours)",
                              "knn_k_sensitivity.png")

        best_k = int(df_knn_sens["RMSE"].idxmin())
        print(f"\nBest k={best_k}. Training final kNN on full data ...")
        knn = SurpriseKNN(k=best_k, sim_name="pearson", user_based=False)
        knn.fit(train)
        results[f"KNN (k={best_k})"] = knn.evaluate(test, label=f"KNN k={best_k}")

        # Top-K ranking metrics
        sample_u = test["user_idx"].unique()[:500]
        ts = test[test["user_idx"].isin(sample_u)].copy()
        ts["pred_rating"] = knn.predict_df(ts)
        ts = ts.rename(columns={"rating": "true_rating"})
        print("\nkNN Top-10 ranking metrics:")
        topk_knn = topk_metrics(ts, k=10, threshold=4.0)
        pd.DataFrame([topk_knn]).to_csv("results/knn_topk_metrics.csv", index=False)

        print(f"kNN total time: {(time.time()-t0)/60:.1f} min")
    else:
        print("\n[STEP 3] kNN skipped (--no-knn)")

    # ── 4. Matrix Factorization ───────────────────────────────────────
    section("STEP 4: Matrix Factorization")
    t0 = time.time()

    # d sweep on 50% subsample, 30 epochs
    print("Running d sensitivity sweep (50% subsample, 30 epochs) ...")
    df_mf_d = mf_sensitivity(
        train, test,
        d_values=[10, 20, 50, 100, 150, 200],
        seed=SEED
    )
    df_mf_d.to_csv("results/mf_d_sensitivity.csv")
    print(df_mf_d)
    save_sensitivity_plot(df_mf_d,
                          "MF Latent Dimension d",
                          "d (latent factors)",
                          "mf_d_sensitivity.png")

    best_d = int(df_mf_d["RMSE"].idxmin())
    print(f"\nBest d={best_d}")

    # lambda sweep using best_d, 50% subsample, 30 epochs
    print(f"\nRunning lambda sensitivity sweep (50% subsample, d={best_d}) ...")
    df_mf_lam = mf_lambda_sensitivity(
        train, test,
        lambda_values=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        best_d=best_d,
        seed=SEED
    )
    df_mf_lam.to_csv("results/mf_lambda_sensitivity.csv")
    print(df_mf_lam)
    save_sensitivity_plot(df_mf_lam,
                          "MF Regularisation Lambda",
                          "lambda",
                          "mf_lambda_sensitivity.png")

    best_lam = float(df_mf_lam["RMSE"].idxmin())
    print(f"\nBest d={best_d}, best lambda={best_lam}")
    print("Training final MF on full data ...")
    mf = SurpriseMF(d=best_d, n_epochs=20, lr_all=0.005, reg_all=best_lam)
    mf.fit(train)
    results[f"MF (d={best_d})"] = mf.evaluate(test, label=f"MF d={best_d}")

    # Top-K ranking metrics
    sample_u = test["user_idx"].unique()[:500]
    ts_mf = test[test["user_idx"].isin(sample_u)].copy()
    ts_mf["pred_rating"] = mf.predict_df(ts_mf)
    ts_mf = ts_mf.rename(columns={"rating": "true_rating"})
    print("\nMF Top-10 ranking metrics:")
    topk_mf = topk_metrics(ts_mf, k=10, threshold=4.0)
    pd.DataFrame([topk_mf]).to_csv("results/mf_topk_metrics.csv", index=False)

    print(f"MF total time: {(time.time()-t0)/60:.1f} min")

    # ── 4b. Confusion matrices (same test subset for all models) ─────
    section("STEP 4b: Confusion matrices (binary like / dislike + MF half-stars)")
    cm_threshold = 4.0
    cm_max_rows = min(150_000, len(test))
    cm_sub = (
        test
        if len(test) <= cm_max_rows
        else test.sample(cm_max_rows, random_state=SEED)
    )
    y_true_cm = cm_sub["rating"].values

    named_binary = [
        ("GlobalMean", binary_like_confusion_matrix(y_true_cm, gm.predict(cm_sub), cm_threshold)),
        ("UserMean", binary_like_confusion_matrix(y_true_cm, um.predict(cm_sub), cm_threshold)),
        ("ItemMean", binary_like_confusion_matrix(y_true_cm, im.predict(cm_sub), cm_threshold)),
        ("BiasModel", binary_like_confusion_matrix(y_true_cm, bm.predict(cm_sub), cm_threshold)),
    ]
    if knn is not None:
        named_binary.append(
            (
                f"KNN (k={knn.k})",
                binary_like_confusion_matrix(y_true_cm, knn.predict_df(cm_sub), cm_threshold),
            )
        )
    named_binary.append(
        (
            f"MF (d={best_d})",
            binary_like_confusion_matrix(y_true_cm, mf.predict_df(cm_sub), cm_threshold),
        )
    )

    save_all_binary_confusion_grid(
        named_binary,
        "results/cm_binary_all_models.png",
        threshold=cm_threshold,
    )
    print("  Saved results/cm_binary_all_models.png")

    summary_rows = []
    for name, cm in named_binary:
        save_binary_confusion_figure(
            cm,
            f"{name}\n(binary, n={len(cm_sub):,})",
            f"results/cm_binary_{_safe_filename_fragment(name)}.png",
            threshold=cm_threshold,
        )
        row = {"model": name, "n_pairs": len(cm_sub)}
        row.update(binary_rates_from_confusion(cm))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv("results/confusion_binary_summary.csv", index=False)
    print("  Saved per-model binary CM PNG/CSV and results/confusion_binary_summary.csv")

    cm_hs, labels_hs = half_star_confusion_matrix(y_true_cm, mf.predict_df(cm_sub))
    save_halfstar_confusion_figure(
        cm_hs,
        labels_hs,
        f"MF (d={best_d}) — half-star buckets (n={len(cm_sub):,})",
        "results/cm_halfstar_mf.png",
    )
    print("  Saved results/cm_halfstar_mf.png (+ .csv)")

    # ── 5. Final comparison ───────────────────────────────────────────
    section("STEP 5: Final results")

    df_all = pd.DataFrame(results).T
    print(df_all.to_string())
    df_all.to_csv("results/all_model_results.csv")

    n_models = len(df_all)
    colors   = (["#4C72B0"] * 4 + ["#DD8452", "#55A868"])[:n_models]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_all["RMSE"].plot(kind="bar", ax=axes[0], color=colors, edgecolor="white")
    axes[0].set_title("Test RMSE - All Models", fontsize=13)
    axes[0].set_ylabel("RMSE (lower is better)")
    axes[0].set_xticklabels(df_all.index, rotation=35, ha="right")
    for i, v in enumerate(df_all["RMSE"]):
        axes[0].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=8)

    df_all["MAE"].plot(kind="bar", ax=axes[1], color=colors, edgecolor="white")
    axes[1].set_title("Test MAE - All Models", fontsize=13)
    axes[1].set_ylabel("MAE (lower is better)")
    axes[1].set_xticklabels(df_all.index, rotation=35, ha="right")
    for i, v in enumerate(df_all["MAE"]):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot saved: results/model_comparison.png")

    # ── 6. Qualitative recommendations ───────────────────────────────
    section("STEP 6: Sample Top-10 Recommendations (MF)")

    sample_uid   = train["str_userId"].iloc[0]
    rated_str    = set(train[train["str_userId"] == sample_uid]["str_movieId"])
    all_str_mids = train["str_movieId"].unique()
    top10        = get_topk_recs_mf(mf, sample_uid, all_str_mids, rated_str, k=10)

    print(f"\nTop-10 for user {sample_uid} "
          f"(rated {len(rated_str)} movies in training):\n")
    for rank, (mid, score) in enumerate(top10, 1):
        t     = movies[movies["movieId"] == int(mid)]["title"].values
        title = t[0] if len(t) > 0 else mid
        print(f"  {rank:>2}. {title:<55} {score:.2f}")

    total_mins = (time.time() - t_start) / 60
    print(f"\nTotal runtime: {total_mins:.1f} minutes")
    print("All results saved to results/")


if __name__ == "__main__":
    main()