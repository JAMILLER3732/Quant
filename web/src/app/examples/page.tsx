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

// The example dataset is the actual portfolio holdings provided for this
// project — 82 real securities, real ticker symbols, real daily closing
// prices (2025) — not placeholder/synthetic data. "SPTR Index" is the one
// column worth overriding after upload: it's a benchmark index level, not a
// tradeable security, so the generic wide-format auto-mapper (which correctly
// defaults unlabelled numeric columns to "close") would otherwise lump it in
// with the holdings.
const ROLE_OVERRIDES = { "SPTR Index": "benchmark" };

// Must match app/quant/portfolio.py's PORTFOLIO_LABEL exactly — selecting
// this value tells any single-security method to analyze an equal-weight,
// daily-rebalanced blend of every mapped holding instead of one stock.
const WHOLE_PORTFOLIO = "◆ Whole Portfolio (all holdings, equal-weight, daily-rebalanced)";

// One curated parameter set per registered method. Every method that can
// meaningfully run on either one security or the whole portfolio defaults
// to the whole portfolio here — this page is meant to demonstrate portfolio
// analysis, not single-stock analysis. The exceptions are structural:
// pairs_trading needs exactly two distinct securities by definition, and
// performance_dashboard / correlation_analysis / efficient_frontier /
// hrp_allocation / stress_testing already operate across every holding at
// once with no "security" selector to begin with.
const METHOD_PARAMS: Record<string, Record<string, unknown>> = {
  returns_descriptive: { security: WHOLE_PORTFOLIO },
  ewma_crossover: { security: WHOLE_PORTFOLIO },
  rolling_zscore: { security: WHOLE_PORTFOLIO },
  performance_dashboard: {},
  monte_carlo_gbm: { security: WHOLE_PORTFOLIO, n_sims: 800 },
  efficient_frontier: { n_random_portfolios: 1500 },
  correlation_analysis: {},
  var_cvar: { security: WHOLE_PORTFOLIO },
  stress_testing: { scenario: "custom_uniform", shock_pct: -20 },
  mean_reversion_backtest: { security: WHOLE_PORTFOLIO },
  factor_analysis: { security: WHOLE_PORTFOLIO },
  pairs_trading: { security_a: "MSFT", security_b: "GOOG" },
  garch_volatility: { security: WHOLE_PORTFOLIO, forecast_days: 10 },
  regime_analysis: { security: WHOLE_PORTFOLIO },
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
        setStatus("Fetching portfolio data…");
        const csvRes = await fetch("/examples/portfolio_holdings.csv");
        const blob = await csvRes.blob();
        const file = new File([blob], "portfolio_holdings.csv", { type: "text/csv" });

        setStatus("Uploading & validating…");
        const upload = await uploadDataset(file);
        if (cancelled) return;

        // The wide-format auto-mapper already resolved every ticker column to
        // "close" — only the benchmark index column needs overriding.
        setStatus("Mapping columns…");
        await updateMapping(upload.dataset_id, { ...upload.role_map, ...ROLE_OVERRIDES });
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
        using this portfolio&apos;s actual holdings (82 real securities, real daily closing prices) as the example
        dataset. Every method below that can run on the portfolio as a whole (rather than one stock) does —
        an equal-weight, daily-rebalanced blend of all 82 holdings, not a single ticker. Click &quot;What is
        this?&quot; on any card for the methodology, required data, assumptions, and limitations.
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
