"use client";

import { useEffect, useState } from "react";
import { ApiError, calculate, type MethodMetadata, type MethodResult } from "@/lib/api";
import PlotlyChart, { type PlotlyFigure } from "@/components/lab/PlotlyChart";
import MethodInfoPanel from "@/components/lab/MethodInfoPanel";

export default function ExampleMethodCard({
  datasetId,
  method,
  params,
}: {
  datasetId: string;
  method: MethodMetadata;
  params: Record<string, unknown>;
}) {
  const [result, setResult] = useState<MethodResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);

  useEffect(() => {
    let cancelled = false;
    calculate(datasetId, method.id, params)
      .then((r) => {
        if (!cancelled) setResult(r);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Calculation failed.");
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, method.id]);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden">
      <div className="p-4 flex items-center justify-between border-b border-slate-800">
        <div>
          <p className="text-xs uppercase tracking-wide text-emerald-400">{method.category}</p>
          <h3 className="font-semibold text-slate-100">{method.name}</h3>
        </div>
        <button
          onClick={() => setShowInfo((v) => !v)}
          className="text-xs rounded-md border border-slate-700 hover:border-slate-500 px-3 py-1.5 text-slate-300 transition-colors shrink-0"
        >
          {showInfo ? "Hide methodology" : "What is this?"}
        </button>
      </div>

      {showInfo && (
        <div className="p-4 border-b border-slate-800 bg-slate-950/40">
          <MethodInfoPanel method={method} />
        </div>
      )}

      <div className="p-4">
        {error && <p className="text-sm text-red-400">{error}</p>}
        {!error && !result && <div className="h-64 animate-pulse rounded-lg bg-slate-800/50" />}
        {result && (
          <>
            <PlotlyChart figure={result.figure as PlotlyFigure} className="h-[360px]" />
            <dl className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(result.stats).slice(0, 8).map(([k, v]) => (
                <div key={k}>
                  <dt className="text-[10px] text-slate-500">{k}</dt>
                  <dd className="text-sm font-semibold text-slate-100">{v === null ? "—" : String(v)}</dd>
                </div>
              ))}
            </dl>
          </>
        )}
      </div>
    </div>
  );
}
