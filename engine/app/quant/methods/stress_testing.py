from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.data.reshape import ReshapeError, extract_series, price_panel
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme

# Documented, widely-cited peak-to-trough index drawdowns used only as an illustrative
# benchmark-level shock magnitude — never presented as a security-specific prediction.
HISTORICAL_SCENARIOS = {
    "gfc_2008": {"label": "2008 Financial Crisis (S&P 500 peak-to-trough)", "benchmark_shock_pct": -56.8},
    "covid_2020": {"label": "COVID-19 Crash, Feb–Mar 2020 (S&P 500 peak-to-trough)", "benchmark_shock_pct": -33.9},
}


class StressTestingMethod(QuantMethod):
    id = "stress_testing"
    name = "Stress Testing & Scenario Analysis"
    category = "Tail Risk & Stress Testing"
    difficulty = "intermediate"

    description = "Apply a custom or historical-benchmark price shock to your securities and see the resulting portfolio P&L, scaled by historical beta when a benchmark is mapped."
    what_it_calculates = (
        "The hypothetical dollar and percentage impact on a portfolio of your selected securities if a "
        "specified price shock occurred, either applied directly (custom) or scaled per-security by historical "
        "beta to a mapped benchmark (historical scenarios)."
    )
    why_use_it = (
        "Tail-risk analysis: understand concentration and downside exposure under an explicit hypothetical "
        "shock, independent of whether that exact shock appears in the historical sample."
    )
    methodology = (
        "For a directly-specified shock, security $i$'s return under the scenario is exactly the shock you "
        "entered. For a historical-benchmark scenario, security $i$'s scenario return is "
        "$\\hat r_i = \\beta_i \\times \\text{shock}_{benchmark}$, where "
        "$\\beta_i = \\text{Cov}(r_i, r_{benchmark}) / \\text{Var}(r_{benchmark})$ is estimated by OLS over the "
        "overlapping historical sample. Portfolio impact is the weighted sum "
        "$\\sum_i w_i \\hat r_i$, using mapped portfolio weights if available, else equal weighting."
    )
    assumptions = [
        "Historical-benchmark scenarios assume the linear (beta) relationship between each security and the "
        "benchmark, estimated in normal markets, continues to hold during the shock — a well-known limitation "
        "since correlations/betas often shift during real crises.",
        "Weights are taken from mapped Portfolio Weight data if available; otherwise all selected securities "
        "are equally weighted.",
        "This is a static, instantaneous shock — it does not model the path, duration, or any second-order "
        "(liquidity, margin call, contagion) effects of a real crisis.",
    ]
    limitations = [
        "Without a mapped benchmark, historical scenarios cannot be beta-scaled and are not offered — you must "
        "use a custom shock in that case.",
        "The 2008/COVID magnitudes are broad market-index drawdowns, not this dataset's securities' actual "
        "historical performance in those periods.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price"),
    ]
    params = [
        ParamSpec("securities", "Securities (comma-separated, blank = all)", "string", default="",
                   description="Securities to include in the stress test."),
        ParamSpec("scenario", "Scenario", "select", default="custom_uniform",
                   options=[
                       {"value": "custom_uniform", "label": "Custom — same shock for all securities"},
                       {"value": "custom_per_security", "label": "Custom — per-security shocks"},
                       {"value": "gfc_2008", "label": "Historical: 2008 Financial Crisis (needs benchmark)"},
                       {"value": "covid_2020", "label": "Historical: COVID-19 Crash (needs benchmark)"},
                   ],
                   description="How the shock is specified."),
        ParamSpec("shock_pct", "Uniform shock (%)", "float", default=-20.0, min=-95, max=95,
                   description="Applied to every selected security when scenario = custom uniform."),
        ParamSpec("per_security_shocks", "Per-security shocks (TICKER:pct, TICKER:pct)", "string", default="",
                   description="e.g. 'AAPL:-25,MSFT:-10' — used when scenario = custom per-security."),
    ]

    def check_requirements(self, role_map, df):
        base = super().check_requirements(role_map, df)
        roles = set(role_map.values())
        if "close" not in roles and "adj_close" not in roles:
            base.satisfied = False
            base.missing.append("Map at least one column to 'Close' or 'Adjusted Close'.")
        return base

    def calculate(self, df: pd.DataFrame, role_map: dict[str, str], params: dict[str, Any]) -> MethodResult:
        panel, price_role_used = price_panel(df, role_map)
        requested = [s.strip() for s in str(params.get("securities", "")).split(",") if s.strip()]
        securities = [s for s in requested if s in panel.columns] or list(panel.columns)

        scenario = params.get("scenario", "custom_uniform")
        returns = panel[securities].pct_change().dropna(how="all")

        warnings: list[str] = []
        shocks: dict[str, float] = {}

        if scenario in ("gfc_2008", "covid_2020"):
            try:
                benchmark = extract_series(df, role_map, "benchmark").pct_change().dropna()
            except ReshapeError as exc:
                raise ValueError(
                    "The historical scenario requires a mapped Benchmark price series to compute beta — none "
                    "was found. Map a benchmark column, or use a custom shock instead."
                ) from exc
            scenario_info = HISTORICAL_SCENARIOS[scenario]
            benchmark_shock = scenario_info["benchmark_shock_pct"] / 100.0
            for sec in securities:
                aligned = pd.concat([panel[sec].pct_change(), benchmark], axis=1, join="inner").dropna()
                if len(aligned) < 20:
                    warnings.append(f"'{sec}' has too little overlapping history with the benchmark to estimate beta reliably; using beta=1.")
                    beta = 1.0
                else:
                    cov = float(np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])
                    var = float(np.var(aligned.iloc[:, 1], ddof=1))
                    beta = cov / var if var > 0 else 1.0
                shocks[sec] = beta * benchmark_shock
            scenario_label = scenario_info["label"]
        elif scenario == "custom_per_security":
            pairs = [p.strip() for p in str(params.get("per_security_shocks", "")).split(",") if p.strip()]
            parsed = {}
            for p in pairs:
                if ":" not in p:
                    continue
                tick, pct = p.split(":", 1)
                try:
                    parsed[tick.strip()] = float(pct) / 100.0
                except ValueError:
                    continue
            if not parsed:
                raise ValueError("No valid per-security shocks parsed. Use format 'AAPL:-25,MSFT:-10'.")
            for sec in securities:
                shocks[sec] = parsed.get(sec, 0.0)
                if sec not in parsed:
                    warnings.append(f"No shock specified for '{sec}' — assumed 0%.")
            scenario_label = "Custom per-security shock"
        else:
            shock = float(params.get("shock_pct", -20.0)) / 100.0
            for sec in securities:
                shocks[sec] = shock
            scenario_label = f"Uniform shock: {shock*100:.1f}%"

        # A security cannot lose more than 100% of its value — but a
        # beta-scaled historical shock (shock_i = beta_i * benchmark_shock)
        # has no such floor built in, and a high-beta name easily produces
        # beta > 100/56.8 ≈ 1.76 under the 2008 scenario, giving a shock
        # below -100% and a NEGATIVE "shocked price", which is meaningless.
        # Custom per-security shocks (free-text) have the same problem if
        # someone types e.g. "-150". Clip every shock at -100% and flag which
        # securities hit that floor, since it signals the linear model has
        # broken down for that name under this shock rather than silently
        # showing an impossible number.
        clipped = [sec for sec, s in shocks.items() if s < -1.0]
        if clipped:
            shocks = {sec: max(s, -1.0) for sec, s in shocks.items()}
            warnings.append(
                f"{', '.join(clipped)} had an implied shock beyond -100% (a security can't lose more than its "
                "full value) — capped at -100%. This usually means a high beta relative to the benchmark shock; "
                "treat these estimates as a lower bound, not precise."
            )

        # weights: from mapped weight data if available (long format), else equal-weight
        weights = {sec: 1.0 / len(securities) for sec in securities}
        if "weight" in role_map.values():
            try:
                from app.data.reshape import extract_panel
                wpanel = extract_panel(df, role_map, "weight")
                latest_weights = wpanel.iloc[-1]
                total = latest_weights.reindex(securities).fillna(0).sum()
                if total > 0:
                    weights = {sec: float(latest_weights.get(sec, 0.0)) / total for sec in securities}
                else:
                    warnings.append("Mapped weights summed to zero or were missing for selected securities; using equal weighting.")
            except Exception:
                warnings.append("Could not read mapped weight data; using equal weighting.")
        else:
            warnings.append("No Portfolio Weight column mapped — assuming equal weighting across selected securities.")

        rows = []
        portfolio_impact = 0.0
        for sec in securities:
            w = weights[sec]
            shock = shocks[sec]
            contribution = w * shock
            portfolio_impact += contribution
            last_price = float(panel[sec].dropna().iloc[-1]) if panel[sec].notna().any() else None
            rows.append({
                "Security": sec, "Weight (%)": round(w * 100, 2), "Shock (%)": round(shock * 100, 2),
                "Contribution to Portfolio Return (%)": round(contribution * 100, 3),
                "Last Price": round(last_price, 4) if last_price is not None else None,
                "Shocked Price": round(last_price * (1 + shock), 4) if last_price is not None else None,
            })

        rows_sorted = sorted(rows, key=lambda r: r["Contribution to Portfolio Return (%)"])

        # A bar per security is fine for a handful of names but an unreadable
        # wall of overlapping x-axis labels past ~20 (an 81-security stress
        # test, for instance) — show only the worst/best N contributors on the
        # chart itself past that threshold; every security's full numbers
        # remain in series_csv_rows/the table regardless.
        MAX_BARS = 20
        many_securities = len(rows_sorted) > MAX_BARS
        if many_securities:
            n_each = MAX_BARS // 2
            chart_rows = rows_sorted[:n_each] + rows_sorted[-n_each:]
        else:
            chart_rows = rows_sorted

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[r["Security"] for r in chart_rows], y=[r["Contribution to Portfolio Return (%)"] for r in chart_rows],
            marker_color=["#dc2626" if v < 0 else "#059669" for v in [r["Contribution to Portfolio Return (%)"] for r in chart_rows]],
            name="Contribution to portfolio return",
        ))
        subtitle = f"Portfolio impact: {portfolio_impact*100:.2f}%"
        if many_securities:
            subtitle += f" — showing the {n_each} best/worst of {len(rows_sorted)} securities; full breakdown in the table below"
        apply_theme(fig, preset=params.get("theme", "professional"),
                    title=f"Stress Test: {scenario_label}", subtitle=subtitle,
                    x_title="Security", y_title="Contribution to Portfolio Return (%)")
        fig.update_xaxes(tickangle=-45 if many_securities else 0)

        stats = {
            "Scenario": scenario_label,
            "Securities Stressed": len(securities),
            "Portfolio Impact (%)": round(portfolio_impact * 100, 2),
            "Worst Contributor": rows_sorted[0]["Security"] if rows_sorted else None,
            "Worst Contribution (%)": rows_sorted[0]["Contribution to Portfolio Return (%)"] if rows_sorted else None,
            "Recovery Required to Break Even (%)": round((1 / (1 + portfolio_impact) - 1) * 100, 2) if (1 + portfolio_impact) != 0 else None,
        }

        if price_role_used == "close":
            warnings.append("Using unadjusted Close prices.")

        return MethodResult(
            figure=self.fig_to_dict(fig),
            stats=stats,
            tables={},
            series_csv_rows=rows,
            warnings=warnings,
        )
