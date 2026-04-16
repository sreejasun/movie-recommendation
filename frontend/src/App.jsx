import { useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import UserInput from "./components/UserInput";
import RecommendationTable from "./components/RecommendationTable";
import MetricsCards from "./components/MetricsCards";
import VisualizationPanel from "./components/VisualizationPanel";
import AboutPanel from "./components/AboutPanel";
import { fetchMetrics, fetchRecommendations } from "./api";

export default function App() {
  const [userId, setUserId] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadMetrics() {
      setLoadingMetrics(true);
      try {
        const response = await fetchMetrics();
        const nextMetrics = response?.metrics || response;
        setMetrics(nextMetrics);
      } catch {
        setError("Unable to load metrics right now. Please check backend service.");
      } finally {
        setLoadingMetrics(false);
      }
    }

    loadMetrics();
  }, []);

  const onGetRecommendations = async () => {
    setError("");
    if (!userId.trim()) {
      setError("Please enter a valid user ID.");
      return;
    }

    setLoadingRecs(true);
    try {
      const response = await fetchRecommendations(userId.trim());
      const list = response?.recommendations || response?.results || [];
      if (!list.length) {
        setRecommendations([]);
        setError("User not found or no recommendations available.");
      } else {
        setRecommendations(list.slice(0, 10));
      }
    } catch {
      setRecommendations([]);
      setError("Could not fetch recommendations. Verify user ID and backend endpoint.");
    } finally {
      setLoadingRecs(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-100 via-slate-100 to-slate-200 dark:from-slate-950 dark:via-slate-950 dark:to-slate-900">
      <Navbar />
      <main className="mx-auto max-w-7xl space-y-6 px-6 py-8">
        <Routes>
          <Route
            path="/"
            element={
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
                <div className="lg:col-span-4">
                  <UserInput
                    userId={userId}
                    setUserId={setUserId}
                    onSubmit={onGetRecommendations}
                    isLoading={loadingRecs}
                    error={error}
                  />
                </div>
                <div className="lg:col-span-8">
                  <RecommendationTable
                    recommendations={recommendations}
                    isLoading={loadingRecs}
                  />
                </div>
              </div>
            }
          />
          <Route
            path="/metrics"
            element={
              <section className="space-y-6">
                <MetricsCards metrics={metrics} isLoading={loadingMetrics} />
                <VisualizationPanel />
              </section>
            }
          />
          <Route path="/about" element={<AboutPanel />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
