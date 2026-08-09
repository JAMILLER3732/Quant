"use client";

import type { QualityReport } from "@/lib/api";

const SEVERITY_STYLE: Record<string, string> = {
  error: "border-red-500/40 bg-red-500/10 text-red-300",
  warning: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  info: "border-sky-500/40 bg-sky-500/10 text-sky-300",
};

const SEVERITY_ICON: Record<string, string> = { error: "✕", warning: "!", info: "i" };

export default function QualityReportPanel({ report }: { report: QualityReport }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h3 className="font-semibold text-slate-100">Data Quality Report</h3>
        <div className="flex gap-4 text-xs text-slate-400">
          <span>
            <strong className="text-slate-200">{report.n_rows.toLocaleString()}</strong> rows
          </span>
          <span>
            <strong className="text-slate-200">{report.n_securities}</strong> securities
          </span>
          {report.date_range && (
            <span>
              <strong className="text-slate-200">
                {report.date_range[0]} → {report.date_range[1]}
              </strong>
            </span>
          )}
          {report.inferred_frequency && (
            <span>
              frequency: <strong className="text-slate-200">{report.inferred_frequency}</strong>
            </span>
          )}
        </div>
      </div>

      <div className="space-y-2">
        {report.issues.map((issue, i) => (
          <div
            key={i}
            className={`flex items-start gap-3 rounded-lg border px-3 py-2 text-sm ${SEVERITY_STYLE[issue.severity]}`}
          >
            <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-current text-[10px] font-bold">
              {SEVERITY_ICON[issue.severity]}
            </span>
            <span>{issue.message}</span>
          </div>
        ))}
      </div>

      {report.has_blocking_errors && (
        <p className="mt-4 text-sm text-red-300">
          Resolve the error(s) above (usually by fixing the column mapping) before running a calculation.
        </p>
      )}
    </div>
  );
}
