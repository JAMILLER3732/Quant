"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ApiError,
  calculate,
  getRequirements,
  listMethods,
  updateMapping,
  uploadDataset,
  type MethodMetadata,
  type MethodResult,
  type QualityReport,
  type RequirementCheck,
  type UploadResponse,
} from "@/lib/api";
import FileUpload from "@/components/lab/FileUpload";
import DataPreviewTable from "@/components/lab/DataPreviewTable";
import QualityReportPanel from "@/components/lab/QualityReportPanel";
import ColumnMapper from "@/components/lab/ColumnMapper";
import MethodPicker from "@/components/lab/MethodPicker";
import MethodInfoPanel from "@/components/lab/MethodInfoPanel";
import ParamsForm from "@/components/lab/ParamsForm";
import ResultsView from "@/components/lab/ResultsView";

type Step = "upload" | "map" | "method" | "results";

export default function QuantLabPage() {
  const [step, setStep] = useState<Step>("upload");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [roleMap, setRoleMap] = useState<Record<string, string>>({});
  const [quality, setQuality] = useState<QualityReport | null>(null);

  const [methods, setMethods] = useState<MethodMetadata[]>([]);
  const [selectedMethodId, setSelectedMethodId] = useState<string | null>(null);
  const [requirements, setRequirements] = useState<RequirementCheck | null>(null);
  const [paramValues, setParamValues] = useState<Record<string, unknown>>({});

  const [result, setResult] = useState<MethodResult | null>(null);

  const selectedMethod = useMemo(
    () => methods.find((m) => m.id === selectedMethodId) ?? null,
    [methods, selectedMethodId]
  );

  useEffect(() => {
    listMethods()
      .then((r) => setMethods(r.methods))
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load methods."));
  }, []);

  async function handleFileSelected(file: File) {
    setBusy(true);
    setError(null);
    try {
      const res = await uploadDataset(file);
      setUpload(res);
      setRoleMap(res.role_map);
      setQuality(res.quality_report);
      setStep("map");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleMappingChange(column: string, role: string) {
    if (!upload) return;
    const next = { ...roleMap };
    if (role === "ignore") delete next[column];
    else next[column] = role;
    setRoleMap(next);
    try {
      const res = await updateMapping(upload.dataset_id, next);
      setQuality(res.quality_report);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to update mapping.");
    }
  }

  async function handleSelectMethod(methodId: string) {
    if (!upload) return;
    setSelectedMethodId(methodId);
    setResult(null);
    setError(null);
    const method = methods.find((m) => m.id === methodId);
    const defaults: Record<string, unknown> = { theme: "professional" };
    method?.params.forEach((p) => (defaults[p.name] = p.default));
    setParamValues(defaults);
    try {
      const req = await getRequirements(upload.dataset_id, methodId);
      setRequirements(req);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to check requirements.");
    }
  }

  async function handleCalculate() {
    if (!upload || !selectedMethodId) return;
    setBusy(true);
    setError(null);
    try {
      const res = await calculate(upload.dataset_id, selectedMethodId, paramValues);
      setResult(res);
      setStep("results");
    } catch (e) {
      if (e instanceof ApiError) {
        const detail = e.detail;
        const missing = typeof detail === "object" ? detail.missing : undefined;
        setError(missing?.length ? `${e.message}: ${missing.join(" ")}` : e.message);
      } else {
        setError("Calculation failed.");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <Stepper step={step} hasUpload={!!upload} hasMethod={!!selectedMethodId} hasResult={!!result} onJump={setStep} />

      {error && (
        <div className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {step === "upload" && (
        <div className="mt-8 max-w-2xl mx-auto">
          <h1 className="text-2xl font-semibold text-slate-100 mb-2">Upload portfolio or market data</h1>
          <p className="text-slate-400 mb-6">
            Upload an Excel or CSV file with dates and prices/returns for one or more securities. The engine will
            inspect the structure, guess column roles, and run a data-quality check before you pick a method.
          </p>
          <FileUpload onFileSelected={handleFileSelected} busy={busy} />
        </div>
      )}

      {step === "map" && upload && (
        <div className="mt-8 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold text-slate-100">
                {upload.filename} — {upload.structure_guess.label}
              </h1>
              <p className="text-slate-400 text-sm mt-1">{upload.structure_guess.description}</p>
            </div>
            <button
              disabled={quality?.has_blocking_errors}
              onClick={() => setStep("method")}
              className="rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed px-4 py-2 text-sm font-medium text-white transition-colors"
            >
              Continue to method selection →
            </button>
          </div>

          {quality && <QualityReportPanel report={quality} />}
          <ColumnMapper columnGuesses={upload.column_guesses} roleMap={roleMap} onChange={handleMappingChange} />

          <div>
            <h3 className="font-semibold text-slate-100 mb-2">Preview (first 25 rows)</h3>
            <DataPreviewTable columns={upload.columns} rows={upload.preview_rows} />
          </div>
        </div>
      )}

      {step === "method" && upload && (
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-6">
          <MethodPicker methods={methods} selectedId={selectedMethodId} onSelect={handleSelectMethod} />

          <div className="space-y-4">
            {!selectedMethod && (
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-8 text-center text-slate-400">
                Select a method from the library to see what it needs and configure it.
              </div>
            )}

            {selectedMethod && (
              <>
                <MethodInfoPanel method={selectedMethod} />

                {requirements && !requirements.satisfied && (
                  <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-5">
                    <p className="font-medium text-red-300 mb-2">
                      This dataset doesn&apos;t have what &quot;{selectedMethod.name}&quot; needs:
                    </p>
                    <ul className="list-disc list-inside text-sm text-red-200 space-y-1">
                      {requirements.missing.map((m, i) => (
                        <li key={i}>{m}</li>
                      ))}
                    </ul>
                    <p className="text-sm text-red-300 mt-2">
                      Go back and adjust the column mapping, or choose a different method.
                    </p>
                  </div>
                )}

                {requirements && requirements.satisfied && (
                  <>
                    <ParamsForm
                      params={selectedMethod.params}
                      values={paramValues}
                      onChange={(name, value) => setParamValues((prev) => ({ ...prev, [name]: value }))}
                      securityOptions={requirements.dynamic_param_options.security ?? []}
                    />
                    <button
                      onClick={handleCalculate}
                      disabled={busy}
                      className="rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-5 py-2.5 text-sm font-medium text-white transition-colors"
                    >
                      {busy ? "Calculating…" : "Calculate & generate chart"}
                    </button>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {step === "results" && result && selectedMethod && (
        <div className="mt-8 space-y-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-semibold text-slate-100">{selectedMethod.name} — Results</h1>
            <button
              onClick={() => setStep("method")}
              className="rounded-md border border-slate-700 hover:border-slate-500 px-4 py-2 text-sm font-medium text-slate-200 transition-colors"
            >
              ← Adjust parameters
            </button>
          </div>
          <ResultsView result={result} methodId={selectedMethod.id} />
        </div>
      )}
    </div>
  );
}

function Stepper({
  step,
  hasUpload,
  hasMethod,
  hasResult,
  onJump,
}: {
  step: Step;
  hasUpload: boolean;
  hasMethod: boolean;
  hasResult: boolean;
  onJump: (s: Step) => void;
}) {
  const steps: { id: Step; label: string; enabled: boolean }[] = [
    { id: "upload", label: "1. Upload", enabled: true },
    { id: "map", label: "2. Validate & Map", enabled: hasUpload },
    { id: "method", label: "3. Select Method", enabled: hasUpload },
    { id: "results", label: "4. Results", enabled: hasResult && hasMethod },
  ];
  return (
    <div className="flex flex-wrap gap-2">
      {steps.map((s) => (
        <button
          key={s.id}
          disabled={!s.enabled}
          onClick={() => onJump(s.id)}
          className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
            step === s.id
              ? "bg-emerald-600 text-white"
              : s.enabled
              ? "bg-slate-800 text-slate-300 hover:bg-slate-700"
              : "bg-slate-900 text-slate-600 cursor-not-allowed"
          }`}
        >
          {s.label}
        </button>
      ))}
    </div>
  );
}
