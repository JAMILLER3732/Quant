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


class ReturnsDescriptiveMethod(QuantMethod):
    id = "returns_descriptive"
    name = "Returns & Descriptive Statistics"
    category = "Returns & Descriptive Statistics"
    difficulty = "beginner"

    description = "Simple/log returns, cumulative return, rolling volatility, and the distributional shape of returns for one security."
    what_it_calculates = (
        "Period-over-period returns, the compounded growth of $1, a rolling annualized volatility estimate, "
        "and distribution diagnostics (skewness, excess kurtosis, histogram) for a single price series."
    )
    why_use_it = (
        "This is the starting point for almost any quant analysis: understanding the return series' shape, "
        "typical magnitude, and tail behavior before applying any risk or optimization model."
    )
    methodology = (
        "Simple return: $r_t = P_t / P_{t-1} - 1$. Log return: $r_t = \\ln(P_t / P_{t-1})$. "
        "Cumulative return curve is the compounded product $\\prod(1+r_t) - 1$. "
        "Rolling volatility uses a trailing window standard deviation of simple returns, annualized by "
        "$\\sigma_{ann} = \\sigma_{daily} \\times \\sqrt{252}$. Skewness and excess kurtosis are the standard "
        "third/fourth standardized moments (excess kurtosis: normal distribution = 0)."
    )
    assumptions = [
        "Returns are calculated from the mapped Close/Adjusted Close price column; if only unadjusted Close is "
        "available, dividend/split effects are not removed.",
        "252 trading days/year is used for annualization (standard equity convention).",
        "Rolling volatility assumes stationarity within each rolling window.",
    ]
    limitations = [
        "Descriptive statistics summarize historical behavior only — they are not a forecast of future returns.",
        "Skewness/kurtosis estimates are noisy with fewer than ~250 observations.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price",
                       note="Map at least one price column (Close or Adjusted Close)."),
    ]
    params = [
        ParamSpec("security", "Security", "select", default=None,
                  description="Which security's price series to analyze."),
        ParamSpec("return_type", "Return type", "select", default="simple",
                  options=[{"value": "simple", "label": "Simple returns"}, {"value": "log", "label": "Log returns"}],
                  description="Simple returns compound multiplicatively; log returns are additive over time."),
        ParamSpec("rolling_window", "Rolling volatility window (days)", "int", default=21, min=5, max=252,
                  description="Trailing window used for the rolling annualized volatility chart."),
    ]

    def check_requirements(self, role_map, df):
        base = super().check_requirements(role_map, df)
        roles = set(role_map.values())
        if "close" not in roles and "adj_close" not in roles:
            base.satisfied = False
            base.missing.append(
                "Map at least one column to 'Close' or 'Adjusted Close' to compute returns."
            )
        return base

    def calculate(self, df: pd.DataFrame, role_map: dict[str, str], params: dict[str, Any]) -> MethodResult:
        panel, price_role_used = price_panel(df, role_map)
        warnings: list[str] = []
        if price_role_used == "close" and "adj_close" not in role_map.values():
            warnings.append(
                "Using unadjusted Close prices — returns may be distorted around dividends/splits."
            )

        prices, security = resolve_security_series(panel, params.get("security"))
        if len(prices) < 3:
            raise ValueError(f"Not enough observations for '{security}' (need at least 3, found {len(prices)}).")

        return_type = params.get("return_type", "simple")
        window = int(params.get("rolling_window", 21))

        returns = calc.log_returns(prices) if return_type == "log" else calc.simple_returns(prices)
        cum = calc.cumulative_returns(calc.simple_returns(prices))  # cumulative growth always on simple returns
        rolling_vol = returns.rolling(window).std(ddof=1) * np.sqrt(calc.TRADING_DAYS_PER_YEAR)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values * 100, name="Cumulative Return (%)",
                                  mode="lines", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values * 100,
                                  name=f"{window}-day Rolling Ann. Volatility (%)", mode="lines",
                                  yaxis="y2", line=dict(width=1.5, dash="dot")))
        fig.update_layout(
            yaxis=dict(title="Cumulative Return (%)"),
            yaxis2=dict(title="Annualized Volatility (%)", overlaying="y", side="right", showgrid=False),
        )
        apply_theme(fig, preset=params.get("theme", "professional"),
                    title=f"{security} — Cumulative Return & Rolling Volatility",
                    subtitle=f"{return_type.title()} returns, {window}-day rolling window",
                    x_title="Date")

        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=returns.values * 100, nbinsx=50, name="Return distribution"))
        apply_theme(hist_fig, preset=params.get("theme", "professional"),
                    title=f"{security} — Return Distribution", x_title="Return (%)", y_title="Frequency", height=360)

        n = len(returns)
        total_return = float(cum.iloc[-1]) if len(cum) else float("nan")
        stats = {
            "Observations": n,
            "Total Return (%)": round(total_return * 100, 2),
            "Annualized Return / CAGR (%)": round(calc.annualize_return(total_return, len(prices) - 1) * 100, 2),
            "Annualized Volatility (%)": round(calc.annualize_vol(returns) * 100, 2),
            "Mean Daily Return (%)": round(float(returns.mean()) * 100, 4),
            "Median Daily Return (%)": round(float(returns.median()) * 100, 4),
            "Skewness": round(calc.skewness(returns), 3),
            "Excess Kurtosis": round(calc.kurtosis(returns), 3),
            "Best Day (%)": round(float(returns.max()) * 100, 2),
            "Worst Day (%)": round(float(returns.min()) * 100, 2),
            "% Positive Days": round(float((returns > 0).mean()) * 100, 1),
        }

        rows = [
            {"date": str(d.date()), "price": float(prices.get(d, np.nan)),
             "return": float(returns.get(d, np.nan)) if d in returns.index else None,
             "cumulative_return": float(cum.get(d, np.nan)) if d in cum.index else None,
             "rolling_volatility": float(rolling_vol.get(d, np.nan)) if d in rolling_vol.index else None}
            for d in prices.index
        ]

        return MethodResult(
            figure=self.fig_to_dict(fig),
            stats=stats,
            tables={"return_histogram": self.fig_to_dict(hist_fig)},
            series_csv_rows=rows,
            warnings=warnings,
        )
