from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.data.reshape import ReshapeError, price_panel
from app.quant import calc
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme
from app.quant.portfolio import (
    efficient_frontier as compute_efficient_frontier,
    mean_returns_and_cov,
    optimize_max_sharpe,
    optimize_min_volatility,
    portfolio_return,
    portfolio_vol,
    simulate_random_portfolios,
)


class EfficientFrontierMethod(QuantMethod):
    id = "efficient_frontier"
    name = "Efficient Frontier & Portfolio Optimization"
    category = "Portfolio Optimization"
    difficulty = "advanced"

    description = "Mean-variance optimization: random-portfolio cloud, the efficient frontier, and the minimum-volatility / maximum-Sharpe portfolios for your mapped securities."
    what_it_calculates = (
        "Expected return, volatility, and Sharpe ratio for thousands of randomly-weighted portfolios of your "
        "mapped securities, plus the mathematically optimal minimum-volatility and maximum-Sharpe portfolios "
        "and the efficient frontier connecting them, via Markowitz mean-variance optimization."
    )
    why_use_it = (
        "The foundational tool for allocating capital across multiple securities to get the best risk-adjusted "
        "return achievable from their historical risk/return/correlation characteristics."
    )
    methodology = (
        "Portfolio return: $E[R_p] = w^T \\mu$. Portfolio variance: $\\sigma_p^2 = w^T \\Sigma w$, where $\\mu$ is "
        "the vector of annualized mean returns and $\\Sigma$ the annualized covariance matrix, estimated from "
        "historical daily returns. The minimum-volatility and maximum-Sharpe portfolios are found via "
        "constrained numerical optimization (SciPy SLSQP) subject to $\\sum w_i = 1$ and (by default) $w_i \\ge 0$. "
        "The efficient frontier traces the minimum-volatility portfolio at each target return level."
    )
    assumptions = [
        "Expected returns and covariance are estimated from historical daily returns and annualized (252 "
        "trading days) — this is the standard, but historically-unstable, MPT input.",
        "Long-only by default (weights bounded [0,1]); short-selling and custom weight bounds are optional.",
        "No rebalancing costs, taxes, or liquidity constraints are modeled.",
    ]
    limitations = [
        "Mean-variance optimization is highly sensitive to the input expected-return estimates — small changes "
        "can produce very different 'optimal' weights (a well-known critique of naive MPT).",
        "Only the overlapping date range across all selected securities is used; short overlap windows produce "
        "unreliable covariance estimates.",
        "This is a historical-data-driven optimization, not investment advice, and says nothing about whether "
        "past risk/return relationships will hold going forward.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price", min_series=2,
                       note="At least 2 securities are required to build a frontier."),
    ]
    params = [
        ParamSpec("securities", "Securities (comma-separated, blank = all)", "string", default="",
                   description="Restrict the optimization to a subset of mapped securities."),
        ParamSpec("risk_free_rate", "Risk-free rate (annualized, %)", "float", default=0.0, min=-5, max=20,
                   description="Used for the Sharpe ratio and max-Sharpe portfolio."),
        ParamSpec("n_random_portfolios", "Random portfolios to simulate", "int", default=4000, min=200, max=20000,
                   description="Size of the random-portfolio cloud shown behind the frontier."),
        ParamSpec("allow_short", "Allow short selling", "bool", default=False,
                   description="If off, all weights are constrained to [0, 1] (long-only)."),
        ParamSpec("max_weight", "Maximum weight per security", "float", default=1.0, min=0.05, max=1.0,
                   description="Upper bound on any single security's weight (long-only mode)."),
        ParamSpec("seed", "Random seed", "int", default=7, min=0, max=2**31 - 1, description="For the random-portfolio cloud."),
    ]

    def check_requirements(self, role_map, df):
        base = super().check_requirements(role_map, df)
        try:
            panel, _ = price_panel(df, role_map)
            if panel.shape[1] < 2:
                base.satisfied = False
                base.missing.append(
                    f"Found only {panel.shape[1]} security with a mapped price series — efficient-frontier "
                    "optimization needs at least 2."
                )
        except ReshapeError:
            base.satisfied = False
            base.missing.append("Map at least 2 securities' Close/Adjusted Close prices.")
        return base

    def calculate(self, df: pd.DataFrame, role_map: dict[str, str], params: dict[str, Any]) -> MethodResult:
        panel, price_role_used = price_panel(df, role_map)

        requested = [s.strip() for s in str(params.get("securities", "")).split(",") if s.strip()]
        securities = [s for s in requested if s in panel.columns] or list(panel.columns)
        if len(securities) < 2:
            raise ValueError(f"Need at least 2 valid securities; got {securities}.")

        returns = panel[securities].pct_change().dropna(how="any")
        if len(returns) < 30:
            raise ValueError(f"Only {len(returns)} overlapping observations across the selected securities — need at least 30.")

        mean_ret, cov = mean_returns_and_cov(returns)
        rf = float(params.get("risk_free_rate", 0.0)) / 100.0
        allow_short = bool(params.get("allow_short", False))
        max_weight = float(params.get("max_weight", 1.0))
        n_random = min(int(params.get("n_random_portfolios", 4000)), 20000)
        seed = int(params.get("seed", 7))

        cloud = simulate_random_portfolios(mean_ret, cov, n_portfolios=n_random, risk_free_rate=rf,
                                            seed=seed, allow_short=allow_short)

        min_vol_w = optimize_min_volatility(mean_ret, cov, allow_short=allow_short, max_weight=max_weight)
        max_sharpe_w = optimize_max_sharpe(mean_ret, cov, risk_free_rate=rf, allow_short=allow_short, max_weight=max_weight)
        frontier = compute_efficient_frontier(mean_ret, cov, allow_short=allow_short, max_weight=max_weight)

        min_vol_ret, min_vol_vol = portfolio_return(min_vol_w, mean_ret), portfolio_vol(min_vol_w, cov)
        max_sharpe_ret, max_sharpe_vol = portfolio_return(max_sharpe_w, mean_ret), portfolio_vol(max_sharpe_w, cov)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cloud["volatility"] * 100, y=cloud["return"] * 100, mode="markers",
                                  marker=dict(size=3, color=cloud["sharpe"], colorscale="Viridis", showscale=True,
                                              colorbar=dict(title="Sharpe")),
                                  name="Random portfolios", hoverinfo="skip"))
        if not frontier.empty:
            fig.add_trace(go.Scatter(x=frontier["volatility"] * 100, y=frontier["target_return"] * 100,
                                      mode="lines", name="Efficient frontier", line=dict(width=3, color="#111827")))
        fig.add_trace(go.Scatter(x=[min_vol_vol * 100], y=[min_vol_ret * 100], mode="markers", name="Min Volatility",
                                  marker=dict(size=14, symbol="star", color="#059669")))
        fig.add_trace(go.Scatter(x=[max_sharpe_vol * 100], y=[max_sharpe_ret * 100], mode="markers", name="Max Sharpe",
                                  marker=dict(size=14, symbol="star", color="#dc2626")))
        apply_theme(fig, preset=params.get("theme", "professional"),
                    title="Efficient Frontier", subtitle=f"{len(securities)} securities, {n_random:,} random portfolios",
                    x_title="Annualized Volatility (%)", y_title="Annualized Expected Return (%)", height=520)

        weights_table = []
        for name, w in [("Minimum Volatility", min_vol_w), ("Maximum Sharpe", max_sharpe_w)]:
            row = {"Portfolio": name,
                   "Expected Return (%)": round(portfolio_return(w, mean_ret) * 100, 2),
                   "Volatility (%)": round(portfolio_vol(w, cov) * 100, 2),
                   "Sharpe": round(portfolio_return(w, mean_ret) / portfolio_vol(w, cov) if portfolio_vol(w, cov) else float("nan"), 2)}
            for sec, weight in zip(securities, w):
                row[f"w_{sec} (%)"] = round(float(weight) * 100, 2)
            weights_table.append(row)

        corr = returns.corr()
        stats = {
            "Securities": ", ".join(securities),
            "Observations Used": len(returns),
            "Min-Vol Portfolio Return (%)": round(min_vol_ret * 100, 2),
            "Min-Vol Portfolio Volatility (%)": round(min_vol_vol * 100, 2),
            "Max-Sharpe Portfolio Return (%)": round(max_sharpe_ret * 100, 2),
            "Max-Sharpe Portfolio Volatility (%)": round(max_sharpe_vol * 100, 2),
            "Max-Sharpe Ratio": round((max_sharpe_ret - rf) / max_sharpe_vol, 2) if max_sharpe_vol else None,
            "Average Pairwise Correlation": round(float((corr.values.sum() - len(securities)) / (len(securities) ** 2 - len(securities))), 3),
        }

        return MethodResult(
            figure=self.fig_to_dict(fig),
            stats=stats,
            tables={"optimal_portfolios": weights_table},
            series_csv_rows=cloud.to_dict(orient="records"),
            warnings=(["Using unadjusted Close prices for at least one security."] if price_role_used == "close" else []),
            notes=[f"Random-portfolio cloud generated with seed={seed} for reproducibility."],
        )
