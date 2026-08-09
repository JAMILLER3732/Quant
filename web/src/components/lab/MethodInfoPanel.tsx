"use client";

import type { MethodMetadata } from "@/lib/api";
import { roleLabel } from "@/lib/roles";

export default function MethodInfoPanel({ method }: { method: MethodMetadata }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4 text-sm">
      <div>
        <h3 className="font-semibold text-slate-100 text-base">{method.name}</h3>
        <p className="text-slate-400 mt-1">{method.description}</p>
      </div>

      <Section title="What it calculates">{method.what_it_calculates}</Section>
      <Section title="Why use it">{method.why_use_it}</Section>

      <div>
        <p className="font-medium text-slate-200 mb-1">Required data</p>
        <ul className="list-disc list-inside space-y-0.5 text-slate-400">
          {method.required_inputs.map((r) => (
            <li key={r.role}>
              {roleLabel(r.role)}
              {r.min_series > 1 ? ` (at least ${r.min_series} securities)` : ""}
              {r.note ? ` — ${r.note}` : ""}
            </li>
          ))}
        </ul>
      </div>

      <Section title="Methodology">{method.methodology}</Section>

      <div>
        <p className="font-medium text-slate-200 mb-1">Assumptions</p>
        <ul className="list-disc list-inside space-y-0.5 text-slate-400">
          {method.assumptions.map((a, i) => (
            <li key={i}>{a}</li>
          ))}
        </ul>
      </div>

      <div>
        <p className="font-medium text-amber-300 mb-1">Limitations</p>
        <ul className="list-disc list-inside space-y-0.5 text-slate-400">
          {method.limitations.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: string }) {
  return (
    <div>
      <p className="font-medium text-slate-200 mb-1">{title}</p>
      <p className="text-slate-400 whitespace-pre-line">{children}</p>
    </div>
  );
}
