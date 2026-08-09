"use client";

import { useMemo, useState } from "react";
import type { MethodMetadata } from "@/lib/api";

export default function MethodPicker({
  methods,
  selectedId,
  onSelect,
}: {
  methods: MethodMetadata[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const [query, setQuery] = useState("");

  const grouped = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = q
      ? methods.filter(
          (m) =>
            m.name.toLowerCase().includes(q) ||
            m.category.toLowerCase().includes(q) ||
            m.description.toLowerCase().includes(q)
        )
      : methods;
    const byCategory = new Map<string, MethodMetadata[]>();
    for (const m of filtered) {
      const list = byCategory.get(m.category) ?? [];
      list.push(m);
      byCategory.set(m.category, list);
    }
    return byCategory;
  }, [methods, query]);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
      <h3 className="font-semibold text-slate-100 mb-3">Quant Method Library</h3>
      <input
        type="text"
        placeholder="Search methods (e.g. volatility, EWMA, Sharpe)…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 mb-4 focus:border-emerald-400 focus:outline-none"
      />
      <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-1">
        {[...grouped.entries()].map(([category, list]) => (
          <div key={category}>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">{category}</p>
            <div className="space-y-1.5">
              {list.map((m) => (
                <button
                  key={m.id}
                  onClick={() => onSelect(m.id)}
                  className={`w-full text-left rounded-lg border px-3 py-2 text-sm transition-colors ${
                    selectedId === m.id
                      ? "border-emerald-400 bg-emerald-400/10 text-emerald-100"
                      : "border-slate-800 hover:border-slate-600 text-slate-200"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium">{m.name}</span>
                    <span className="text-[10px] uppercase tracking-wide text-slate-500">{m.difficulty}</span>
                  </div>
                  <p className="mt-0.5 text-xs text-slate-400 line-clamp-2">{m.description}</p>
                </button>
              ))}
            </div>
          </div>
        ))}
        {grouped.size === 0 && <p className="text-sm text-slate-500">No methods match &quot;{query}&quot;.</p>}
      </div>
    </div>
  );
}
