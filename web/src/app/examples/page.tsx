"use client";

import { useEffect, useState } from "react";
import {
  ApiError,
  listMethods,
  updateMapping,
  uploadDataset,
  type MethodMetadata,
} from "@/lib/api";
import ExampleMethodCard from "@/components/examples/ExampleMethodCard";

const ROLE_MAP = {
  Date: "date",
  EXMP_TECH: "close",
  EXMP_STAPLE: "close",
  EXMP_ENERGY: "close",
  EXMP_HEALTH: "close",
  EXMP_BENCHMARK: "benchmark",
};

// One curated parameter set per registered method, tuned against the bundled
// synthetic example dataset (5 securities, ~3 years of daily data) so every
// method in the library has a working, representative showcase.
const METHOD_PARAMS: Record<string, Record<string, unknown>> = {
  returns_descriptive: { security: "EXMP_TECH" },
  ewma_crossover: { security: "EXMP_TECH" },
  rolling_zscore: { security: "EXMP_TECH" },
  performance_dashboard: {},
  monte_carlo_gbm: { security: "EXMP_TECH", n_sims: 800 },
  efficient_frontier: { n_random_portfolios: 1500 },
  correlation_analysis: {},
  var_cvar: { security: "EXMP_TECH" },
  stress_testing: { scenario: "custom_uniform", shock_pct: -20 },
  mean_reversion_backtest: { security: "EXMP_STAPLE" },
  factor_analysis: { security: "EXMP_TECH" },
  pairs_trading: { security_a: "EXMP_STAPLE", security_b: "EXMP_HEALTH" },
  garch_volatility: { security: "EXMP_TECH", forecast_days: 10 },
  regime_analysis: { security: "EXMP_TECH" },
  hrp_allocation: {},
};

export default function ExamplesPage() {
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [methods, setMethods] = useState<MethodMetadata[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState("Loading example dataset…");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setStatus("Fetching example data…");
        const csvRes = await fetch("/examples/sample_portfolio.csv");
        const blob = await csvRes.blob();
        const file = new File([blob], "sample_portfolio.csv", { type: "text/csv" });

        setStatus("Uploading & validating…");
        const upload = await uploadDataset(file);
        if (cancelled) return;

        setStatus("Mapping columns…");
        await updateMapping(upload.dataset_id, ROLE_MAP);
        if (cancelled) return;

        setStatus("Loading method library…");
        const methodsRes = await listMethods();
        if (cancelled) return;

        setMethods(methodsRes.methods);
        setDatasetId(upload.dataset_id);
      } catch (e) {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Failed to load examples.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <p className="text-sm font-medium text-emerald-400 mb-2">Method Library — Examples</p>
      <h1 className="text-2xl font-semibold text-slate-100 mb-2">
        See every method in action, before you upload
      </h1>
      <p className="text-slate-400 mb-8 max-w-3xl">
        Every chart below is calculated live by the same Python quant engine your own uploads run through —
        using a bundled <strong>synthetic</strong> 5-security example dataset (clearly-labeled placeholder
        tickers, ~3 years of simulated daily prices), not real market data. Click &quot;What is this?&quot; on
        any card for the methodology, required data, assumptions, and limitations.
      </p>

      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300 mb-6">
          {error}
        </div>
      )}

      {!datasetId && !error && (
        <div className="flex items-center gap-3 text-slate-400">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-600 border-t-emerald-400" />
          {status}
        </div>
      )}

      {datasetId && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {methods.map((method) => (
            <ExampleMethodCard
              key={method.id}
              datasetId={datasetId}
              method={method}
              params={METHOD_PARAMS[method.id] ?? {}}
            />
          ))}
        </div>
      )}
    </div>
  );
}
