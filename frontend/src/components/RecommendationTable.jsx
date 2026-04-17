import { Sparkles } from "lucide-react";
import SkeletonRows from "./SkeletonRows";

function modelLabel(model) {
  if (!model) return null;
  if (model.engine === "mf" || model.name === "matrix_factorization") {
    const d = model.latent_factors ?? model.d;
    return `Matrix factorization (SVD) · d=${d ?? "—"}`;
  }
  if (model.engine === "bias" || model.name === "bias_model") {
    return "Bias model (μ + user/item biases)";
  }
  return null;
}

export default function RecommendationTable({ recommendations, model, isLoading }) {
  if (isLoading) {
    return (
      <section className="glass-panel animate-floatIn p-6">
        <div className="mb-4 flex items-center gap-2 text-slate-600 dark:text-slate-300">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-500/30 border-t-slate-200" />
          Fetching personalized recommendations...
        </div>
        <SkeletonRows rows={6} />
      </section>
    );
  }

  if (!recommendations?.length) {
    return (
      <section className="glass-panel animate-floatIn p-10 text-center">
        <Sparkles className="mx-auto mb-3 text-brand-400" size={22} />
        <h3 className="text-base font-semibold text-slate-900 dark:text-white">No recommendations yet</h3>
        <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
          Search with a valid user ID to see top 10 movie recommendations.
        </p>
      </section>
    );
  }

  return (
    <section className="glass-panel animate-floatIn overflow-hidden">
      <div className="border-b border-slate-200 px-6 py-4 dark:border-slate-800">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Top 10 Recommendations</h2>
          {modelLabel(model) ? (
            <span className="rounded-full border border-brand-500/40 bg-brand-500/15 px-3 py-1 text-xs font-medium text-red-600 dark:text-red-300">
              {modelLabel(model)}
            </span>
          ) : null}
        </div>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Sorted by predicted rating (training-set movies excluded)
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-100 text-xs uppercase tracking-wide text-slate-500 dark:bg-slate-900/80 dark:text-slate-400">
            <tr>
              <th className="px-6 py-3">#</th>
              <th className="px-6 py-3">Movie Title</th>
              <th className="px-6 py-3 text-right">Predicted Rating</th>
            </tr>
          </thead>
          <tbody>
            {recommendations.map((movie, index) => (
              <tr
                key={`${movie.title}-${index}`}
                className="border-t border-slate-200 text-slate-700 transition hover:bg-slate-100 dark:border-slate-800/70 dark:text-slate-200 dark:hover:bg-slate-800/45"
              >
                <td className="px-6 py-4 text-slate-500 dark:text-slate-400">{index + 1}</td>
                <td className="px-6 py-4 font-medium">{movie.title}</td>
                <td className="px-6 py-4 text-right font-semibold text-emerald-400">
                  {Number(movie.predicted_rating).toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
