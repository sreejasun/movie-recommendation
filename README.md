# Movie Recommendation using Collaborative Filtering & Matrix Factorization

---

## Team Members

- Sreeja Sunkeswaram
- Hemalasya Annapureddy
- Chitra Chowdary Joguparthi
- Hemanth Reddy Gundepally
- Sai Nithin Reddy Ponna

---

## Project Structure

```
movie_recommendation/
├── data/                        ← Put your Kaggle CSVs here
│   ├── rating.csv
│   ├── movie.csv
│   ├── tag.csv
│   ├── genome_scores.csv
│   ├── genome_tags.csv
│   ├── train.parquet            ← Auto-generated after first run
│   └── test.parquet             ← Auto-generated after first run
│
├── src/                         ← All reusable Python modules
│   ├── preprocessing.py         ← Load, clean, encode, split
│   ├── evaluation.py            ← RMSE, MAE, Precision@K, Recall@K, NDCG@K
│   ├── baselines.py             ← GlobalMean, UserMean, ItemMean, BiasModel
│   ├── knn_cf.py                ← Item-item kNN (surprise wrapper + sensitivity)
│   └── matrix_factorization.py ← From-scratch MF + SurpriseMF + sensitivity
│
├── notebooks/
│   ├── 01_eda.ipynb             ← EDA, rating distribution, sparsity, split
│   ├── 02_baselines.ipynb       ← Fit & evaluate all baselines
│   ├── 03_knn.ipynb             ← kNN experiments, k sensitivity, Top-K metrics
│   ├── 04_mf.ipynb              ← MF experiments, d & λ ablations, Top-K metrics
│   └── 05_evaluation.ipynb      ← Final comparison, error analysis, report plots
│
├── results/                     ← Auto-generated: CSVs + PNG plots
│   ├── all_model_results.csv    ← Master comparison table (all models)
│   ├── knn_sensitivity.csv      ← k vs RMSE/MAE
│   ├── mf_d_sensitivity.csv     ← d vs RMSE/MAE
│   └── model_comparison.png     ← Bar chart comparing all models
│
├── report/                      ← Place your final report PDF here
├── run_all.py                   ← Single-script end-to-end pipeline
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Download the dataset

Go to https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset  
Download and extract the archive into the `data/` folder.

### 3a. Run via single script

```bash
# Quick dev run with 10% of data (recommended first):
python run_all.py --data-dir data/ --sample 0.1

# Full 20M row run:
python run_all.py --data-dir data/

# Skip kNN (faster):
python run_all.py --data-dir data/ --sample 0.1 --no-knn
```

### 3b. Run via Jupyter Notebooks (step by step)

```bash
jupyter notebook
```

Open the `notebooks/` folder and run in order: **01 → 02 → 03 → 04 → 05**

> Always run `01_eda.ipynb` first — it generates `train.parquet` and `test.parquet` which all other notebooks depend on.

---

## Methods

### Baseline Models (`src/baselines.py`)

| Model       | Formula                                 |
| ----------- | --------------------------------------- |
| Global Mean | r̂ = μ                                   |
| User Mean   | r̂ = μ_u                                 |
| Item Mean   | r̂ = μ_i                                 |
| Bias Model  | r̂ = μ + b_u + b_i (SGD, L2 regularised) |

### Neighbourhood Collaborative Filtering (`src/knn_cf.py`)

Item-item kNN with Pearson similarity:

```
r̂_ui = μ_u + Σ_{j ∈ N_k(i;u)} sim(i,j) · (r_uj − μ_j)
               ─────────────────────────────────────────
                      Σ |sim(i,j)|
```

- Shrinkage via `min_support` threshold
- Sensitivity sweep over k ∈ {10, 20, 40, 60, 80, 100}

### Matrix Factorization (`src/matrix_factorization.py`)

```
r̂_ui = μ + b_u + b_i + p_u^T q_i

min  Σ (r_ui − r̂_ui)² + λ (‖p_u‖² + ‖q_i‖² + b_u² + b_i²)
 Ω
```

- Trained with SGD
- Ablations over d ∈ {10, 20, 50, 100, 150, 200} and λ ∈ {0.001 … 0.2}

---

## Evaluation

**Rating prediction:**

- RMSE = √(1/|Ω_test| · Σ (r_ui − r̂_ui)²)
- MAE = 1/|Ω_test| · Σ |r_ui − r̂_ui|

**Top-K ranking** (threshold = 4.0 stars as relevant):

- Precision@K, Recall@K, NDCG@K

**Error analysis:**

- Per user-activity bucket (≤10, 11–50, 51–200, >200 training ratings)

---

## Results

After running the pipeline, all results are saved to `results/`:

| File                    | Description                          |
| ----------------------- | ------------------------------------ |
| `all_model_results.csv` | RMSE and MAE for every model         |
| `model_comparison.png`  | Bar chart of all models              |
| `knn_sensitivity.csv`   | Effect of varying k on RMSE/MAE      |
| `mf_d_sensitivity.csv`  | Effect of varying latent dimension d |

---

## Tips

- Use `--sample 0.1` while developing — runs in minutes instead of hours.
- `train.parquet` and `test.parquet` are saved to `data/` after the first run, so subsequent runs skip the slow split step automatically.
- All plots are saved as 150-dpi PNGs, ready to use in your report.

---

## References

1. Koren, Bell, Volinsky (2009). _Matrix Factorization Techniques for Recommender Systems._ IEEE Computer.
2. Yao et al. (2014). _Dual-Regularized One-Class Collaborative Filtering._ CIKM.
3. Kaggle MovieLens 20M Dataset — https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
