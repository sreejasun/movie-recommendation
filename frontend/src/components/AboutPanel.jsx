export default function AboutPanel() {
  return (
    <section className="glass-panel animate-floatIn p-6">
      <h2 className="text-xl font-semibold text-slate-900 dark:text-white">About This Project</h2>
      <p className="mt-3 text-sm leading-6 text-slate-700 dark:text-slate-300">
        This dashboard presents personalized movie recommendations powered by collaborative
        filtering and matrix factorization. It highlights predicted top movies, key model
        metrics (RMSE/MAE), and evaluation visuals such as model comparison and confusion
        matrix plots.
      </p>
      <div className="mt-5 rounded-xl border border-slate-300 bg-slate-100 p-4 text-sm text-slate-600 dark:border-slate-800 dark:bg-slate-900/70 dark:text-slate-400">
        Tip: start with a known user ID from your dataset to get meaningful recommendations.
      </div>
    </section>
  );
}
