"use client";

import { useEffect, useState } from "react";
import { ApiError, downloadReportPdf, getAiStatus, previewReport, type ReportOptions, type ReportPreview } from "@/lib/api";

export default function ReportPanel({
  datasetId,
  securityOptions,
}: {
  datasetId: string;
  securityOptions: string[];
}) {
  const [scope, setScope] = useState<"portfolio" | "security">(securityOptions.length > 1 ? "portfolio" : "security");
  const [security, setSecurity] = useState(securityOptions[0] ?? "");
  const [includeOptimization, setIncludeOptimization] = useState(false);
  const [aiConfigured, setAiConfigured] = useState<boolean | null>(null);
  const [aiModel, setAiModel] = useState<string | null>(null);
  const [preview, setPreview] = useState<ReportPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAiStatus()
      .then((s) => {
        setAiConfigured(s.configured);
        setAiModel(s.model);
      })
      .catch(() => setAiConfigured(false));
  }, []);

  const options: ReportOptions = {
    scope,
    security: scope === "security" ? security : undefined,
    include_optimization: includeOptimization,
  };

  async function handlePreview() {
    setBusy(true);
    setError(null);
    setPreview(null);
    try {
      const result = await previewReport(datasetId, options);
      setPreview(result);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Report generation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleDownload() {
    setDownloading(true);
    setError(null);
    try {
      const hint = scope === "security" ? `report_${security}` : "portfolio_report";
      await downloadReportPdf(datasetId, options, hint);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "PDF download failed.");
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
        <h3 className="font-semibold text-slate-100 mb-1">Analysis Report</h3>
        <p className="text-sm text-slate-400 mb-4">
          Generates an institutional-style research note — real computed statistics and charts, with
          {aiConfigured ? " AI-written" : " templated"} narrative sections. Downloadable as PDF.
        </p>

        {aiConfigured === false && (
          <div className="mb-4 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs text-sky-300">
            AI narrative isn&apos;t configured on this deployment — the report still generates in full, with
            plain templated narrative instead of AI-written prose.
          </div>
        )}
        {aiConfigured && (
          <div className="mb-4 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-300">
            AI narrative enabled ({aiModel}) — grounded strictly in the computed figures below.
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-slate-400">Scope</label>
            <select
              value={scope}
              onChange={(e) => setScope(e.target.value as "portfolio" | "security")}
              className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm text-slate-100 focus:border-emerald-400 focus:outline-none"
            >
              <option value="portfolio">Full Portfolio</option>
              <option value="security">Single Security</option>
            </select>
          </div>
          {scope === "security" && (
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-slate-400">Security</label>
              <select
                value={security}
                onChange={(e) => setSecurity(e.target.value)}
                className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm text-slate-100 focus:border-emerald-400 focus:outline-none"
              >
                {securityOptions.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
          )}
          {scope === "portfolio" && (
            <div className="flex items-center gap-2 pt-5">
              <input
                type="checkbox"
                checked={includeOptimization}
                onChange={(e) => setIncludeOptimization(e.target.checked)}
                className="h-4 w-4 rounded border-slate-700 bg-slate-950 accent-emerald-500"
              />
              <label className="text-xs font-medium text-slate-400">Include efficient frontier (slower)</label>
            </div>
          )}
        </div>

        <div className="flex gap-3">
          <button
            onClick={handlePreview}
            disabled={busy || (scope === "security" && !security)}
            className="rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-4 py-2 text-sm font-medium text-white transition-colors"
          >
            {busy ? "Generating…" : "Preview Report"}
          </button>
          <button
            onClick={handleDownload}
            disabled={downloading || (scope === "security" && !security)}
            className="rounded-md border border-slate-700 hover:border-slate-500 disabled:opacity-50 px-4 py-2 text-sm font-medium text-slate-200 transition-colors"
          >
            {downloading ? "Preparing PDF…" : "Download PDF"}
          </button>
        </div>

        {error && (
          <div className="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}
      </div>

      {preview && (
        <div className="rounded-xl border border-slate-800 bg-white overflow-hidden">
          <iframe
            title="Report preview"
            srcDoc={`<!DOCTYPE html><html><head><meta charset="utf-8"/></head><body>${preview.html}</body></html>`}
            className="w-full"
            style={{ height: "80vh", border: "none" }}
          />
        </div>
      )}
    </div>
  );
}
