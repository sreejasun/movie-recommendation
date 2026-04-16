import axios from "axios";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "",
  timeout: 15000
});

export async function fetchRecommendations(userId) {
  const { data } = await client.get("/recommend", {
    params: { user_id: userId }
  });
  return data;
}

export async function fetchMetrics() {
  const { data } = await client.get("/metrics");
  return data;
}
