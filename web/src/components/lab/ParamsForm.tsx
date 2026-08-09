"use client";

import type { ParamSpec } from "@/lib/api";

const THEME_OPTIONS = [
  { value: "professional", label: "Professional" },
  { value: "dark_quant", label: "Dark Quant" },
  { value: "minimal", label: "Minimal" },
  { value: "bloomberg", label: "Bloomberg-inspired" },
  { value: "presentation", label: "Presentation" },
];

export default function ParamsForm({
  params,
  values,
  onChange,
  securityOptions,
}: {
  params: ParamSpec[];
  values: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
  securityOptions: string[];
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
      <h3 className="font-semibold text-slate-100 mb-3">Parameters</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {params.map((p) => (
          <Field key={p.name} spec={p} value={values[p.name]} onChange={(v) => onChange(p.name, v)} securityOptions={securityOptions} />
        ))}

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-slate-400">Chart theme</label>
          <select
            value={(values.theme as string) ?? "professional"}
            onChange={(e) => onChange("theme", e.target.value)}
            className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm text-slate-100 focus:border-emerald-400 focus:outline-none"
          >
            {THEME_OPTIONS.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}

function Field({
  spec,
  value,
  onChange,
  securityOptions,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
  securityOptions: string[];
}) {
  const label = (
    <label className="text-xs font-medium text-slate-400" title={spec.description}>
      {spec.label}
    </label>
  );

  if (spec.name === "security") {
    return (
      <div className="flex flex-col gap-1">
        {label}
        <select
          value={(value as string) ?? securityOptions[0] ?? ""}
          onChange={(e) => onChange(e.target.value)}
          className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm text-slate-100 focus:border-emerald-400 focus:outline-none"
        >
          {securityOptions.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>
    );
  }

  if (spec.type === "select" && spec.options) {
    return (
      <div className="flex flex-col gap-1">
        {label}
        <select
          value={String(value ?? spec.default)}
          onChange={(e) => onChange(e.target.value)}
          className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm text-slate-100 focus:border-emerald-400 focus:outline-none"
        >
          {spec.options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
    );
  }

  if (spec.type === "bool") {
    return (
      <div className="flex items-center gap-2 pt-4">
        <input
          type="checkbox"
          checked={Boolean(value ?? spec.default)}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 rounded border-slate-700 bg-slate-950 accent-emerald-500"
        />
        {label}
      </div>
    );
  }

  if (spec.type === "int" || spec.type === "float") {
    return (
      <div className="flex flex-col gap-1">
        {label}
        <input
          type="number"
          value={(value as number) ?? (spec.default as number)}
          min={spec.min ?? undefined}
          max={spec.max ?? undefined}
          step={spec.step ?? (spec.type === "int" ? 1 : 0.01)}
          onChange={(e) => onChange(spec.type === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
          className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm text-slate-100 focus:border-emerald-400 focus:outline-none"
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      {label}
      <input
        type="text"
        value={(value as string) ?? (spec.default as string) ?? ""}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm text-slate-100 focus:border-emerald-400 focus:outline-none"
      />
    </div>
  );
}
