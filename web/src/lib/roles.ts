// Mirrors app/data/column_detection.py ROLES on the engine — kept in sync manually
// since this is a small, stable list. If the engine adds a role, add it here too.
export const ROLES: { value: string; label: string }[] = [
  { value: "ignore", label: "— Ignore this column —" },
  { value: "date", label: "Date" },
  { value: "ticker", label: "Ticker / Symbol" },
  { value: "open", label: "Open" },
  { value: "high", label: "High" },
  { value: "low", label: "Low" },
  { value: "close", label: "Close" },
  { value: "adj_close", label: "Adjusted Close" },
  { value: "volume", label: "Volume" },
  { value: "returns", label: "Returns" },
  { value: "weight", label: "Portfolio Weight" },
  { value: "position", label: "Position" },
  { value: "quantity", label: "Quantity / Shares" },
  { value: "pnl", label: "P&L" },
  { value: "benchmark", label: "Benchmark" },
  { value: "risk_free", label: "Risk-Free Rate" },
  { value: "portfolio_value", label: "Portfolio Value" },
  { value: "factor", label: "Factor" },
  { value: "sector", label: "Sector / Industry" },
];

export function roleLabel(role: string): string {
  return ROLES.find((r) => r.value === role)?.label ?? role;
}
