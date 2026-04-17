# Frontend Dashboard

React + Tailwind dashboard for the Movie Recommendation System.

## Run locally

1. Install dependencies:

```bash
npm install
```

2. Configure API URL (optional — in dev, the app defaults to `http://127.0.0.1:8000`):

```bash
cp .env.example .env
# edit .env if your API is not on port 8000
```

3. Start development server:

```bash
npm run dev
```

The dev server proxies `/recommend`, `/metrics`, and `/results` to `http://127.0.0.1:8000`.

**Backend first:** start the FastAPI app on port 8000 (`backend/README.md`) so API calls succeed.

## Troubleshooting

- **Do not run `npm audit fix --force`.** It can upgrade Vite to an incompatible major version and break the dev server.
- **Port 5173 busy:** Vite will pick the next free port (for example `5174`); use the URL printed in the terminal.
- **`npm warn Unknown env config "devdir"`:** comes from a global npm config; it does not stop the app from running.
