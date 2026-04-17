#!/usr/bin/env python3
"""
Train Matrix Factorization (Surprise SVD) on data/train.parquet and save to results/mf_model.pkl
for the FastAPI backend to load at startup.

Usage (from project root):
  .venv/bin/python backend/train_mf_server.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.matrix_factorization import SurpriseMF, save_mf_model  # noqa: E402


def main() -> None:
    DATA_DIR = ROOT / "data"
    train_path = DATA_DIR / "train.parquet"
    if not train_path.exists():
        print("Missing data/train.parquet — run run_all.py or 01_eda.ipynb first.", file=sys.stderr)
        sys.exit(1)

    import pandas as pd

    train = pd.read_parquet(train_path)
    if "str_userId" not in train.columns:
        train["str_userId"] = train["userId"].astype(str)
    if "str_movieId" not in train.columns:
        train["str_movieId"] = train["movieId"].astype(str)

    d = int(os.environ.get("MF_D", "10"))
    n_epochs = int(os.environ.get("MF_EPOCHS", "20"))
    lr = float(os.environ.get("MF_LR", "0.005"))
    reg = float(os.environ.get("MF_REG", "0.02"))

    out = ROOT / "results" / "mf_model.pkl"
    print(f"Training SurpriseMF  d={d}  epochs={n_epochs}  lr={lr}  reg={reg}  rows={len(train):,}")
    mf = SurpriseMF(d=d, n_epochs=n_epochs, lr_all=lr, reg_all=reg)
    mf.fit(train)
    save_mf_model(mf, out)
    print(f"Done. Restart the API to load {out}")


if __name__ == "__main__":
    main()
