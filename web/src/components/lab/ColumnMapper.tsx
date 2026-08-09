"use client";

import { ROLES } from "@/lib/roles";
import type { ColumnGuess } from "@/lib/api";

export default function ColumnMapper({
  columnGuesses,
  roleMap,
  onChange,
}: {
  columnGuesses: ColumnGuess[];
  roleMap: Record<string, string>;
  onChange: (column: string, role: string) => void;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
      <h3 className="font-semibold text-slate-100 mb-1">Column Mapping</h3>
      <p className="text-sm text-slate-400 mb-4">
        We guessed a role for each column below. Correct anything that&apos;s wrong before running an analysis —
        calculations use these roles, not the raw column names.
      </p>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="text-slate-400">
            <tr>
              <th className="px-2 py-1.5 text-left font-medium">Column</th>
              <th className="px-2 py-1.5 text-left font-medium">Detected role</th>
              <th className="px-2 py-1.5 text-left font-medium">Confidence</th>
              <th className="px-2 py-1.5 text-left font-medium">Assign role</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {columnGuesses.map((g) => {
              const current = roleMap[g.column] ?? "ignore";
              return (
                <tr key={g.column}>
                  <td className="px-2 py-2 font-medium text-slate-200 whitespace-nowrap">{g.column}</td>
                  <td className="px-2 py-2 text-slate-400 whitespace-nowrap">{g.role}</td>
                  <td className="px-2 py-2 text-slate-500 whitespace-nowrap">{Math.round(g.confidence * 100)}%</td>
                  <td className="px-2 py-2">
                    <select
                      value={current}
                      onChange={(e) => onChange(g.column, e.target.value)}
                      className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100 text-sm focus:border-emerald-400 focus:outline-none"
                    >
                      {ROLES.map((r) => (
                        <option key={r.value} value={r.value}>
                          {r.label}
                        </option>
                      ))}
                    </select>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
