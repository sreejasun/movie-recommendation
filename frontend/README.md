# Frontend Dashboard

React + Tailwind dashboard for the Movie Recommendation System.

## Run locally

1. Install dependencies:

```bash
npm install
```

2. Configure API URL (optional):

```bash
cp .env.example .env
```

3. Start development server:

```bash
npm run dev
```

The dev server proxies `/recommend`, `/metrics`, and `/results` to `http://127.0.0.1:8000`.
