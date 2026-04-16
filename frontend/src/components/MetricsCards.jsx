import { Activity, Gauge } from "lucide-react";
import SkeletonRows from "./SkeletonRows";

function MetricCard({ label, value, icon: Icon }) {
  return (
    <div className="glass-panel flex items-center gap-3 p-5 transition hover:scale-[1.01]">
      <div className="rounded-lg bg-brand-500/20 p-2 text-brand-400">
        <Icon size={18} />
      </div>
      <div>
        <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
        <p className="text-xl font-bold text-slate-900 dark:text-white">{value}</p>
      </div>
    </div>
  );
}

export default function MetricsCards({ metrics, isLoading }) {
  return (
    <section className="space-y-4">
      <div className="flex items-end justify-between">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Model Performance</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400">Test-set quality indicators</p>
      </div>

      {isLoading ? (
        <div className="glass-panel">
          <div className="px-5 pt-5 text-sm text-slate-600 dark:text-slate-300">Loading metrics...</div>
          <SkeletonRows rows={2} />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <MetricCard
            label="RMSE"
            value={metrics?.rmse ? Number(metrics.rmse).toFixed(4) : "--"}
            icon={Gauge}
          />
          <MetricCard
            label="MAE"
            value={metrics?.mae ? Number(metrics.mae).toFixed(4) : "--"}
            icon={Activity}
          />
        </div>
      )}
    </section>
  );
}
