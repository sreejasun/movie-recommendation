import { Search, User } from "lucide-react";

export default function UserInput({
  userId,
  setUserId,
  onSubmit,
  isLoading,
  error
}) {
  return (
    <section className="glass-panel animate-floatIn p-6">
      <h2 className="mb-1 text-lg font-semibold text-slate-900 dark:text-white">Find Recommendations</h2>
      <p className="mb-5 text-sm text-slate-500 dark:text-slate-400">
        Enter a user ID to get top personalized movie suggestions.
      </p>

      <div className="space-y-4">
        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">User ID</label>
        <div className="relative">
          <User
            size={16}
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 dark:text-slate-500"
          />
          <input
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="e.g. 123"
            className="w-full rounded-xl border border-slate-300 bg-white px-10 py-2.5 text-sm text-slate-900 outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-500/30 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
          />
        </div>

        <button
          type="button"
          onClick={onSubmit}
          disabled={isLoading}
          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-brand-500 px-4 py-2.5 text-sm font-semibold text-white transition hover:scale-[1.01] hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-70"
        >
          {isLoading ? (
            <>
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              Loading...
            </>
          ) : (
            <>
              <Search size={16} />
              Get Recommendations
            </>
          )}
        </button>
      </div>

      {error ? (
        <p className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
          {error}
        </p>
      ) : null}
    </section>
  );
}
