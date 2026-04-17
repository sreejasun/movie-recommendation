import axios from "axios";

/**
 * Browser → FastAPI. In dev, default to direct URL so we don't depend only on the Vite proxy.
 * Set VITE_API_BASE_URL in .env to override (e.g. production API).
 */
export function apiBaseURL() {
  const fromEnv = import.meta.env.VITE_API_BASE_URL;
  if (fromEnv !== undefined && fromEnv !== "") {
    return fromEnv.replace(/\/$/, "");
  }
  if (import.meta.env.DEV) {
    return "http://127.0.0.1:8000";
  }
  return "";
}

const client = axios.create({
  baseURL: apiBaseURL(),
  timeout: 60000
});

function normalizeAxiosError(error) {
  if (!axios.isAxiosError(error)) {
    return error instanceof Error ? error.message : String(error);
  }
  const status = error.response?.status;
  const detail = error.response?.data?.detail;
  if (status === 404) {
    if (typeof detail === "string") {
      return detail;
    }
    return "User not found in the training data.";
  }
  if (status === 422 && Array.isArray(detail)) {
    const msg = detail.map((d) => d.msg || d).join("; ");
    return msg || "Invalid request.";
  }
  if (error.code === "ECONNREFUSED" || error.message?.includes("Network Error")) {
    return "Cannot reach the API. Start the backend: .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000";
  }
  if (error.code === "ECONNABORTED") {
    return "Request timed out. The backend may still be loading data (first start can take a minute).";
  }
  if (typeof detail === "string") {
    return detail;
  }
  return error.message || "Request failed.";
}

export async function fetchRecommendations(userId) {
  const uid = typeof userId === "string" ? userId.trim() : userId;
  try {
    const { data } = await client.get("/recommend", {
      params: { user_id: uid }
    });
    return data;
  } catch (e) {
    throw new Error(normalizeAxiosError(e));
  }
}

export async function fetchMetrics() {
  try {
    const { data } = await client.get("/metrics");
    return data;
  } catch (e) {
    throw new Error(normalizeAxiosError(e));
  }
}
