"use client";

import type { MethodResult } from "@/lib/api";
import { downloadTextFile, rowsToCsv } from "@/lib/csv";
import PlotlyChart, { type PlotlyFigure } from "./PlotlyChart";

function isFigure(v: unknown): v is PlotlyFigure {
  return typeof v === "object" && v !== null && "data" in v;
}

export default function ResultsView({ result, methodId }: { result: MethodResult; methodId: string }) {
  const secondaryFigures = Object.entries(result.tables).filter(([, v]) => isFigure(v)) as [string, PlotlyFigure][];
  const secondaryTables = Object.entries(result.tables).filter(
    ([, v]) => Array.isArray(v) && v.length > 0 && !isFigure(v)
  ) as [string, Record<string, unknown>[]][];

  return (
    <div className="space-y-6">
      {result.warnings.length > 0 && (
        <div className="space-y-1.5">
          {result.warnings.map((w, i) => (
            <div
              key={i}
              className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-300"
            >
              ⚠ {w}
            </div>
          ))}
        </div>
      )}

      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-slate-100">Chart</h3>
          <button
            onClick={() =>
              downloadTextFile(`${methodId}_results.csv`, rowsToCsv(result.csv_rows), "text/csv")
            }
            className="rounded-md bg-emerald-600 hover:bg-emerald-500 px-3 py-1.5 text-xs font-medium text-white transition-colors"
          >
            Download data (CSV)
          </button>
        </div>
        <PlotlyChart figure={result.figure as PlotlyFigure} filenameForExport={methodId} className="h-[480px]" />
        <p className="mt-2 text-xs text-slate-500">
          Use the camera icon in the chart toolbar to export PNG/SVG at high resolution.
        </p>
      </div>

      {secondaryFigures.map(([name, fig]) => (
        <div key={name} className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <h3 className="font-semibold text-slate-100 mb-2 capitalize">{name.replace(/_/g, " ")}</h3>
          <PlotlyChart figure={fig} filenameForExport={`${methodId}_${name}`} className="h-[380px]" />
        </div>
      ))}

      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
        <h3 className="font-semibold text-slate-100 mb-3">Statistics</h3>
        <dl className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          {Object.entries(result.stats).map(([k, v]) => (
            <div key={k}>
              <dt className="text-xs text-slate-500">{k}</dt>
              <dd className="text-lg font-semibold text-slate-100">{v === null || v === undefined ? "—" : String(v)}</dd>
            </div>
          ))}
        </dl>
      </div>

      {secondaryTables.map(([name, rows]) => (
        <div key={name} className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold text-slate-100 capitalize">{name.replace(/_/g, " ")}</h3>
            <button
              onClick={() => downloadTextFile(`${methodId}_${name}.csv`, rowsToCsv(rows), "text/csv")}
              className="rounded-md border border-slate-700 hover:border-slate-500 px-3 py-1.5 text-xs font-medium text-slate-200 transition-colors"
            >
              Download CSV
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="text-slate-400">
                <tr>
                  {Object.keys(rows[0]).map((h) => (
                    <th key={h} className="px-2 py-1.5 text-left font-medium whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {rows.map((r, i) => (
                  <tr key={i}>
                    {Object.keys(rows[0]).map((h) => (
                      <td key={h} className="px-2 py-1.5 whitespace-nowrap text-slate-300">
                        {String(r[h])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      {result.notes.length > 0 && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 text-sm text-slate-400">
          {result.notes.map((n, i) => (
            <p key={i}>{n}</p>
          ))}
        </div>
      )}
    </div>
  );
}
