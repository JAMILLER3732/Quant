// Typed client for the Python quant engine API.
// Base URL comes from NEXT_PUBLIC_API_BASE_URL (set per-environment); falls
// back to localhost for local development against `uvicorn app.main:app`.

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export type ColumnGuess = {
  column: string;
  role: string;
  confidence: number;
  reason: string;
};

export type StructureGuess = {
  format_id: string;
  label: string;
  description: string;
  confidence: number;
};

export type QualityIssue = {
  severity: "info" | "warning" | "error";
  code: string;
  message: string;
  detail: Record<string, unknown>;
};

export type QualityReport = {
  n_rows: number;
  n_columns: number;
  n_securities: number;
  date_range: [string, string] | null;
  inferred_frequency: string | null;
  issues: QualityIssue[];
  has_blocking_errors: boolean;
};

export type UploadResponse = {
  dataset_id: string;
  filename: string;
  sheet_names: string[];
  used_sheet: string | null;
  n_rows: number;
  n_columns: number;
  columns: string[];
  column_guesses: ColumnGuess[];
  structure_guess: StructureGuess;
  role_map: Record<string, string>;
  preview_rows: Record<string, unknown>[];
  quality_report: QualityReport;
};

export type ParamSpec = {
  name: string;
  label: string;
  type: "int" | "float" | "select" | "bool" | "string";
  default: unknown;
  min: number | null;
  max: number | null;
  step: number | null;
  options: { value: string; label: string }[] | null;
  description: string;
};

export type RequiredInput = {
  role: string;
  label: string;
  min_series: number;
  required: boolean;
  note: string;
};

export type MethodMetadata = {
  id: string;
  name: string;
  category: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  description: string;
  what_it_calculates: string;
  why_use_it: string;
  methodology: string;
  assumptions: string[];
  limitations: string[];
  required_inputs: RequiredInput[];
  params: ParamSpec[];
};

export type RequirementCheck = {
  method_id: string;
  satisfied: boolean;
  missing: string[];
  warnings: string[];
  dynamic_param_options: Record<string, string[]>;
};

export type MethodResult = {
  method_id: string;
  figure: Record<string, unknown>;
  stats: Record<string, unknown>;
  tables: Record<string, unknown>;
  csv_rows: Record<string, unknown>[];
  warnings: string[];
  notes: string[];
  params_used: Record<string, unknown>;
};

export type ApiErrorDetail = {
  error?: string;
  message?: string;
  missing?: string[];
};

export class ApiError extends Error {
  detail: ApiErrorDetail | string;
  status: number;
  constructor(status: number, detail: ApiErrorDetail | string) {
    const message = typeof detail === "string" ? detail : detail.message || "Request failed";
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
  });
  if (!res.ok) {
    let detail: ApiErrorDetail | string = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body;
    } catch {
      // non-JSON error body
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export async function uploadDataset(file: File, sheetName?: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  if (sheetName) form.append("sheet_name", sheetName);
  const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
  if (!res.ok) {
    let detail: ApiErrorDetail | string = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body;
    } catch {
      // ignore
    }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export function updateMapping(datasetId: string, roleMap: Record<string, string>) {
  return request<{ role_map: Record<string, string>; quality_report: QualityReport }>(
    `/api/datasets/${datasetId}/mapping`,
    { method: "POST", body: JSON.stringify({ role_map: roleMap }) }
  );
}

export function listMethods() {
  return request<{ methods: MethodMetadata[] }>("/api/methods");
}

export function getRequirements(datasetId: string, methodId: string) {
  return request<RequirementCheck>(`/api/datasets/${datasetId}/methods/${methodId}/requirements`);
}

export function calculate(datasetId: string, methodId: string, params: Record<string, unknown>) {
  return request<MethodResult>(`/api/datasets/${datasetId}/calculate/${methodId}`, {
    method: "POST",
    body: JSON.stringify({ params }),
  });
}

export type ReportOptions = {
  scope: "security" | "portfolio";
  security?: string;
  include_optimization?: boolean;
};

export type ReportPreview = {
  html: string;
  title: string;
  ai_generated: boolean;
  sections: string[];
  warnings: string[];
};

export function getAiStatus() {
  return request<{ configured: boolean; model: string | null }>("/api/reports/ai-status");
}

export function previewReport(datasetId: string, options: ReportOptions) {
  return request<ReportPreview>(`/api/datasets/${datasetId}/report/preview`, {
    method: "POST",
    body: JSON.stringify(options),
  });
}

export async function downloadReportPdf(datasetId: string, options: ReportOptions, filenameHint: string) {
  const res = await fetch(`${API_BASE}/api/datasets/${datasetId}/report/pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });
  if (!res.ok) {
    let detail: ApiErrorDetail | string = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body;
    } catch {
      // ignore
    }
    throw new ApiError(res.status, detail);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${filenameHint}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
