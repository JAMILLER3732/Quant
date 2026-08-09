from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.data.reshape import price_panel
from app.quant import calc
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme
from app.quant.portfolio import resolve_security_series
from app.quant.simulation import gbm_paths, percentile_bands

MAX_SIMS = 20000
MAX_HORIZON_DAYS = 252 * 5


class MonteCarloGbmMethod(QuantMethod):
    id = "monte_carlo_gbm"
    name = "Monte Carlo Simulation (Geometric Brownian Motion)"
    category = "Simulation"
    difficulty = "intermediate"

    description = "Simulates thousands of future price paths under GBM using drift/volatility estimated from your data (or set manually), with percentile bands."
    what_it_calculates = (
        "Thousands of possible future price paths for a security, generated under the Geometric Brownian "
        "Motion model, summarized as percentile bands (5th/25th/50th/75th/95th) and a terminal-price "
        "distribution."
    )
    why_use_it = (
        "A standard way to visualize the range of plausible future outcomes and their uncertainty, e.g. for "
        "risk budgeting or scenario planning — explicitly NOT a single-point forecast."
    )
    methodology = (
        "GBM: $dS_t = \\mu S_t \\, dt + \\sigma S_t \\, dW_t$, with exact solution "
        "$S_{t+\\Delta t} = S_t \\exp\\big((\\mu - \\tfrac12\\sigma^2)\\Delta t + \\sigma\\sqrt{\\Delta t}\\, Z\\big)$, "
        "$Z \\sim N(0,1)$. Historical drift/vol (if selected) are estimated as the annualized mean and standard "
        "deviation of simple daily returns. All paths are generated with one vectorized NumPy draw of shape "
        "(n_sims, n_steps) — no per-path loop."
    )
    assumptions = [
        "Returns are lognormally distributed with constant drift and volatility over the simulation horizon — "
        "GBM does not model volatility clustering, jumps, or regime changes.",
        "If historical estimation is used, the future is assumed to statistically resemble the historical "
        "sample window.",
        "No dividends, transaction costs, or market-impact effects are modeled.",
    ]
    limitations = [
        "The median/mean path is NOT a forecast or guarantee — it is the center of a wide, uncertain "
        "distribution. Real markets exhibit fat tails and volatility clustering that GBM does not capture.",
        "Drift is notoriously hard to estimate reliably from historical returns over short windows; results are "
        "highly sensitive to the drift assumption.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price"),
    ]
    params = [
        ParamSpec("security", "Security", "select", default=None, description="Security to simulate."),
        ParamSpec("horizon_days", "Forecast horizon (trading days)", "int", default=252, min=5, max=MAX_HORIZON_DAYS,
                   description="Number of trading days to simulate forward."),
        ParamSpec("n_sims", "Number of simulations", "int", default=2000, min=100, max=MAX_SIMS,
                   description="More simulations -> smoother percentile estimates, more compute."),
        ParamSpec("drift_mode", "Drift (μ)", "select", default="historical",
                   options=[{"value": "historical", "label": "Estimate from historical data"},
                            {"value": "manual", "label": "Set manually"}],
                   description="Annualized expected return used as the drift term."),
        ParamSpec("manual_drift", "Manual annualized drift (%)", "float", default=8.0, min=-50, max=100,
                   description="Used only if drift is set manually."),
        ParamSpec("vol_mode", "Volatility (σ)", "select", default="historical",
                   options=[{"value": "historical", "label": "Estimate from historical data"},
                            {"value": "manual", "label": "Set manually"}],
                   description="Annualized volatility used as the diffusion term."),
        ParamSpec("manual_vol", "Manual annualized volatility (%)", "float", default=25.0, min=1, max=200,
                   description="Used only if volatility is set manually."),
        ParamSpec("seed", "Random seed", "int", default=42, min=0, max=2**31 - 1,
                   description="Fixes the random draw so results are reproducible."),
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
        prices, security = resolve_security_series(panel, params.get("security"))
        if len(prices) < 30:
            raise ValueError(f"Need at least 30 historical observations to estimate drift/volatility (found {len(prices)}).")

        s0 = float(prices.iloc[-1])
        hist_returns = calc.simple_returns(prices)
        hist_drift = calc.annualize_return(float((1 + hist_returns).prod() - 1), len(hist_returns))
        hist_vol = calc.annualize_vol(hist_returns)

        drift_mode = params.get("drift_mode", "historical")
        vol_mode = params.get("vol_mode", "historical")
        mu = hist_drift if drift_mode == "historical" else float(params.get("manual_drift", 8.0)) / 100.0
        sigma = hist_vol if vol_mode == "historical" else float(params.get("manual_vol", 25.0)) / 100.0
        if sigma <= 0:
            raise ValueError("Volatility must be positive for a GBM simulation.")

        horizon_days = min(int(params.get("horizon_days", 252)), MAX_HORIZON_DAYS)
        n_sims = min(int(params.get("n_sims", 2000)), MAX_SIMS)
        seed = int(params.get("seed", 42))

        paths = gbm_paths(s0, mu, sigma, horizon_days, n_sims, seed=seed)
        bands = percentile_bands(paths)
        time_axis = np.arange(horizon_days + 1)

        fig = go.Figure()
        n_sample_paths = min(150, n_sims)
        sample_idx = np.random.default_rng(seed).choice(n_sims, size=n_sample_paths, replace=False)
        for i in sample_idx:
            fig.add_trace(go.Scatter(x=time_axis, y=paths[i], mode="lines", line=dict(width=0.5, color="#94a3b8"),
                                      opacity=0.15, showlegend=False, hoverinfo="skip"))
        colors = {5: "#dc2626", 25: "#f59e0b", 50: "#2563eb", 75: "#f59e0b", 95: "#dc2626"}
        for p in (5, 25, 50, 75, 95):
            fig.add_trace(go.Scatter(x=time_axis, y=bands[p], mode="lines", name=f"P{p}",
                                      line=dict(width=2 if p == 50 else 1.3, color=colors[p],
                                                dash="solid" if p == 50 else "dot")))
        apply_theme(fig, preset=params.get("theme", "professional"),
                    title=f"{security} — Monte Carlo GBM Simulation ({n_sims:,} paths, {horizon_days} trading days)",
                    subtitle=f"μ={mu*100:.1f}%/yr, σ={sigma*100:.1f}%/yr ({drift_mode} drift, {vol_mode} vol) — descriptive scenario range, not a forecast",
                    x_title="Trading Days Ahead", y_title="Simulated Price")

        terminal = paths[:, -1]
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=terminal, nbinsx=60, name="Terminal price distribution"))
        hist_fig.add_vline(x=s0, line=dict(color="#94a3b8", dash="dash"))
        apply_theme(hist_fig, preset=params.get("theme", "professional"),
                    title=f"{security} — Terminal Price Distribution (day {horizon_days})",
                    x_title="Simulated Price", y_title="Frequency", height=340)

        stats = {
            "Current Price": round(s0, 4),
            "Annualized Drift Used (%)": round(mu * 100, 2),
            "Annualized Volatility Used (%)": round(sigma * 100, 2),
            "Historical Drift Estimate (%)": round(hist_drift * 100, 2),
            "Historical Volatility Estimate (%)": round(hist_vol * 100, 2),
            "Simulations": n_sims,
            "Horizon (trading days)": horizon_days,
            "Median Terminal Price": round(float(np.median(terminal)), 4),
            "5th Percentile Terminal Price": round(float(bands[5][-1]), 4),
            "95th Percentile Terminal Price": round(float(bands[95][-1]), 4),
            "P(Terminal > Current Price) (%)": round(float((terminal > s0).mean()) * 100, 1),
            "Random Seed": seed,
        }

        rows = [
            {"day": int(t), "p5": float(bands[5][t]), "p25": float(bands[25][t]), "p50": float(bands[50][t]),
             "p75": float(bands[75][t]), "p95": float(bands[95][t])}
            for t in range(horizon_days + 1)
        ]

        warnings = []
        if price_role_used == "close":
            warnings.append("Using unadjusted Close prices to estimate historical drift/volatility.")
        if drift_mode == "historical" and len(hist_returns) < 252:
            warnings.append(
                f"Historical drift was estimated from only {len(hist_returns)} return observations — "
                "annualized drift estimates are highly noisy over short windows."
            )

        return MethodResult(
            figure=self.fig_to_dict(fig),
            stats=stats,
            tables={"terminal_distribution": self.fig_to_dict(hist_fig)},
            series_csv_rows=rows,
            warnings=warnings,
            notes=["Simulation uses a fixed random seed for reproducibility; rerun with a different seed to see sampling variability."],
        )
