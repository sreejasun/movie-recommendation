## Backend API

FastAPI service for the frontend dashboard.

### Run

```bash
cd /Users/nithinreddy/Documents/movie_recommendation
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

### Matrix Factorization model (recommended)

The API loads **Matrix Factorization** (Surprise **SVD**) from `results/mf_model.pkl` when that file exists.  
If the file is missing, `/recommend` falls back to a bias model (μ + user/item biases).

Generate the MF weights once (uses `data/train.parquet`; can take a while on large data):

```bash
cd /Users/nithinreddy/Documents/movie_recommendation
.venv/bin/python backend/train_mf_server.py
```

Then restart Uvicorn. Optional env vars: `MF_D`, `MF_EPOCHS`, `MF_LR`, `MF_REG` (same as training script).

### Endpoints

- `GET /health`
- `GET /metrics`
- `GET /recommend?user_id=1&k=10`
- `GET /results/<plot-file>`
