import Link from "next/link";

const PILLARS = [
  {
    title: "Upload & Validate",
    body: "Drop in Excel/CSV portfolio or market data. The engine detects structure, guesses column roles, and runs a data-quality report before anything is calculated.",
  },
  {
    title: "Python Quant Engine",
    body: "Every chart is backed by a real calculation — pandas/NumPy/SciPy running documented, testable formulas. No fabricated numbers, ever.",
  },
  {
    title: "Understand, then export",
    body: "Every method explains what it calculates, its assumptions, and its limitations — then lets you download the chart and the underlying results.",
  },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-20 text-center">
      <p className="text-sm font-medium text-emerald-400 mb-3">Quantitative Finance Portfolio Analytics</p>
      <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight text-slate-50">
        Upload data. Select a methodology.
        <br />
        Get mathematically correct charts.
      </h1>
      <p className="mt-5 text-lg text-slate-400 max-w-2xl mx-auto">
        A research workstation for portfolio analytics, risk, optimization, and backtesting — every result is
        calculated from your data by a documented, tested Python quant engine.
      </p>
      <Link
        href="/lab"
        className="mt-8 inline-block rounded-md bg-emerald-600 hover:bg-emerald-500 px-6 py-3 text-sm font-semibold text-white transition-colors"
      >
        Open the Quant Lab →
      </Link>

      <div className="mt-20 grid grid-cols-1 sm:grid-cols-3 gap-6 text-left">
        {PILLARS.map((p) => (
          <div key={p.title} className="rounded-xl border border-slate-800 bg-slate-900/40 p-6">
            <h3 className="font-semibold text-slate-100 mb-2">{p.title}</h3>
            <p className="text-sm text-slate-400">{p.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
