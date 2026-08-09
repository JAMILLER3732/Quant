from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

from app.data.reshape import ReshapeError, extract_series, price_panel
from app.quant import calc
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme
from app.quant.portfolio import resolve_security_series


class FactorAnalysisMethod(QuantMethod):
    id = "factor_analysis"
    name = "Factor Analysis (CAPM / Market Model)"
    category = "Factor Analysis"
    difficulty = "intermediate"

    description = "OLS regression of a security's excess returns on a mapped benchmark's excess returns: alpha, beta, R², and rolling beta/alpha through time."
    what_it_calculates = (
        "The market-model (CAPM) regression $r_i - r_f = \\alpha + \\beta (r_m - r_f) + \\epsilon$: annualized "
        "alpha, market beta, R-squared, regression significance, and a rolling-window beta/alpha time series "
        "showing how the relationship has evolved."
    )
    why_use_it = (
        "Beta quantifies systematic (market) risk exposure; alpha is the average return unexplained by that "
        "exposure. This is the foundational single-factor model underlying more advanced multi-factor analysis."
    )
    methodology = (
        "OLS regression via statsmodels: $r_{i,t} - r_{f,t} = \\alpha + \\beta (r_{m,t} - r_{f,t}) + \\epsilon_t$. "
        "$\\beta = \\text{Cov}(r_i, r_m) / \\text{Var}(r_m)$ is the slope; $\\alpha$ (annualized) is the intercept "
        "scaled by 252. $R^2$ is the fraction of the security's return variance explained by market movements. "
        "Rolling beta/alpha re-fit the same regression on a trailing window."
    )
    assumptions = [
        "Uses the mapped Benchmark as the single market factor; risk-free rate defaults to 0 if not mapped.",
        "OLS assumes linear relationship, homoskedastic errors, and no autocorrelation in residuals — not "
        "formally tested here.",
        "Only the overlapping date range between the security and benchmark is used.",
    ]
    limitations = [
        "A single-factor model explains only market-driven return variation; unexplained residual risk "
        "(captured by 1-R²) can be substantial, especially for individual securities vs. diversified benchmarks.",
        "Beta estimated over one historical window is not guaranteed to be stable going forward.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price"),
        RequiredInput(role="benchmark", label="Benchmark price series", note="Required as the market factor."),
    ]
    params = [
        ParamSpec("security", "Security", "select", default=None, description="Security to regress."),
        ParamSpec("rolling_window", "Rolling window (days)", "int", default=126, min=30, max=504,
                   description="Trailing window for the rolling beta/alpha chart."),
        ParamSpec("risk_free_rate", "Risk-free rate (annualized, %)", "float", default=0.0, min=-5, max=20,
                   description="Used to compute excess returns; uses mapped Risk-Free Rate column if available, else this constant."),
    ]

    def check_requirements(self, role_map, df):
        base = super().check_requirements(role_map, df)
        roles = set(role_map.values())
        if "close" not in roles and "adj_close" not in roles:
            base.satisfied = False
            base.missing.append("Map at least one column to 'Close' or 'Adjusted Close'.")
        if "benchmark" not in roles:
            base.satisfied = False
            base.missing.append("Map a 'Benchmark' price column — factor analysis needs a market factor to regress against.")
        return base

    def calculate(self, df: pd.DataFrame, role_map: dict[str, str], params: dict[str, Any]) -> MethodResult:
        panel, price_role_used = price_panel(df, role_map)
        prices, security = resolve_security_series(panel, params.get("security"))

        try:
            benchmark_prices = extract_series(df, role_map, "benchmark")
        except ReshapeError as exc:
            raise ValueError("Could not read the mapped Benchmark series.") from exc

        sec_returns = calc.simple_returns(prices)
        bench_returns = calc.simple_returns(benchmark_prices.dropna())
        aligned = pd.concat([sec_returns.rename("security"), bench_returns.rename("market")], axis=1).dropna()
        if len(aligned) < 30:
            raise ValueError(f"Only {len(aligned)} overlapping observations between security and benchmark — need at least 30.")

        rf_annual = float(params.get("risk_free_rate", 0.0)) / 100.0
        rf_daily = (1 + rf_annual) ** (1 / calc.TRADING_DAYS_PER_YEAR) - 1

        excess_sec = aligned["security"] - rf_daily
        excess_mkt = aligned["market"] - rf_daily
        X = sm.add_constant(excess_mkt.values)
        model = sm.OLS(excess_sec.values, X).fit()
        alpha_daily, beta = model.params[0], model.params[1]
        alpha_annual = alpha_daily * calc.TRADING_DAYS_PER_YEAR

        window = int(params.get("rolling_window", 126))
        roll_beta, roll_alpha = [], []
        idx = aligned.index
        for i in range(len(aligned)):
            if i < window:
                roll_beta.append(np.nan)
                roll_alpha.append(np.nan)
                continue
            window_slice = aligned.iloc[i - window: i]
            xm = window_slice["market"].values - rf_daily
            ym = window_slice["security"].values - rf_daily
            xw = sm.add_constant(xm)
            m = sm.OLS(ym, xw).fit()
            roll_alpha.append(m.params[0] * calc.TRADING_DAYS_PER_YEAR)
            roll_beta.append(m.params[1])
        roll_beta_s = pd.Series(roll_beta, index=idx)
        roll_alpha_s = pd.Series(roll_alpha, index=idx)

        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(x=excess_mkt.values * 100, y=excess_sec.values * 100, mode="markers",
                                          marker=dict(size=4, opacity=0.5), name="Daily excess returns"))
        line_x = np.linspace(excess_mkt.min(), excess_mkt.max(), 50)
        line_y = alpha_daily + beta * line_x
        scatter_fig.add_trace(go.Scatter(x=line_x * 100, y=line_y * 100, mode="lines", name="OLS fit",
                                          line=dict(color="#dc2626", width=2)))
        apply_theme(scatter_fig, preset=params.get("theme", "professional"),
                    title=f"{security} vs. Benchmark — Market Model", subtitle=f"β={beta:.2f}, α(annual)={alpha_annual*100:.2f}%, R²={model.rsquared:.3f}",
                    x_title="Benchmark Excess Return (%)", y_title=f"{security} Excess Return (%)")

        roll_fig = go.Figure()
        roll_fig.add_trace(go.Scatter(x=roll_beta_s.index, y=roll_beta_s.values, name="Rolling Beta", line=dict(width=1.8)))
        roll_fig.add_hline(y=1, line=dict(color="#94a3b8", dash="dash", width=1))
        apply_theme(roll_fig, preset=params.get("theme", "professional"), title=f"Rolling {window}-day Beta",
                    x_title="Date", y_title="Beta", height=320)

        roll_alpha_fig = go.Figure()
        roll_alpha_fig.add_trace(go.Scatter(x=roll_alpha_s.index, y=roll_alpha_s.values * 100, name="Rolling Alpha (annualized %)",
                                             line=dict(width=1.8, color="#059669")))
        roll_alpha_fig.add_hline(y=0, line=dict(color="#94a3b8", width=1))
        apply_theme(roll_alpha_fig, preset=params.get("theme", "professional"), title=f"Rolling {window}-day Annualized Alpha",
                    x_title="Date", y_title="Alpha (%)", height=320)

        stats = {
            "Beta": round(float(beta), 3),
            "Alpha (annualized, %)": round(float(alpha_annual) * 100, 2),
            "R-squared": round(float(model.rsquared), 3),
            "Beta t-stat": round(float(model.tvalues[1]), 2),
            "Beta p-value": round(float(model.pvalues[1]), 4),
            "Observations": len(aligned),
            "Risk-Free Rate Used (%)": round(rf_annual * 100, 2),
        }

        rows = [
            {"date": str(d.date()), "security_return": float(aligned["security"].get(d, np.nan)),
             "benchmark_return": float(aligned["market"].get(d, np.nan)),
             "rolling_beta": float(roll_beta_s.get(d, np.nan)), "rolling_alpha_annualized": float(roll_alpha_s.get(d, np.nan))}
            for d in aligned.index
        ]

        warnings = []
        if price_role_used == "close":
            warnings.append("Using unadjusted Close prices for at least the security or benchmark.")
        if model.pvalues[1] > 0.05:
            warnings.append(f"Beta is not statistically significant at the 5% level (p={model.pvalues[1]:.3f}) — interpret with caution.")

        return MethodResult(
            figure=self.fig_to_dict(scatter_fig),
            stats=stats,
            tables={"rolling_beta": self.fig_to_dict(roll_fig), "rolling_alpha": self.fig_to_dict(roll_alpha_fig)},
            series_csv_rows=rows,
            warnings=warnings,
        )
