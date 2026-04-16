## Backend API

FastAPI service for the frontend dashboard.

### Run

```bash
cd /Users/nithinreddy/Documents/movie_recommendation
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

### Endpoints

- `GET /health`
- `GET /metrics`
- `GET /recommend?user_id=1&k=10`
- `GET /results/<plot-file>`
