const plots = [
  {
    title: "Model Comparison",
    path: "/results/model_comparison.png"
  },
  {
    title: "Confusion Matrix",
    path: "/results/cm_binary_all_models.png"
  }
];

export default function VisualizationPanel() {
  return (
    <section className="space-y-4">
      <div className="flex items-end justify-between">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Visualizations</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400">Plots served by backend results</p>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {plots.map((plot) => (
          <article
            key={plot.path}
            className="glass-panel overflow-hidden transition hover:shadow-xl hover:shadow-black/20"
          >
            <div className="border-b border-slate-800 px-4 py-3">
              <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">{plot.title}</h3>
            </div>
            <div className="aspect-[16/10] bg-slate-100 dark:bg-slate-950">
              <img
                src={plot.path}
                alt={plot.title}
                className="h-full w-full object-contain p-2"
                loading="lazy"
              />
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
