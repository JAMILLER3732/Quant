from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.data.reshape import price_panel
from app.quant import calc
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme


class RollingZScoreMethod(QuantMethod):
    id = "rolling_zscore"
    name = "Rolling Z-Score & Standard Deviation Bands"
    category = "Returns & Descriptive Statistics"
    difficulty = "beginner"

    description = "Rolling mean/stdev bands on price and the corresponding rolling Z-score of returns — a standard mean-reversion / anomaly-detection view."
    what_it_calculates = (
        "A rolling mean and +/-k standard-deviation envelope around price, and the rolling Z-score of returns "
        "(how many standard deviations the latest return is from its own trailing mean)."
    )
    why_use_it = (
        "Highlights statistically unusual price/return behavior and is a building block for mean-reversion and "
        "pairs-trading signal construction."
    )
    methodology = (
        "Rolling mean/stdev computed over a trailing window of size $w$: "
        "$\\mu_t = \\frac{1}{w}\\sum_{i=t-w+1}^{t} r_i$, $\\sigma_t = \\text{stdev of the same window}$. "
        "Z-score: $z_t = (r_t - \\mu_t) / \\sigma_t$. Bands on the price chart are $\\mu_t \\pm k\\sigma_t$ using "
        "the rolling price mean/stdev directly."
    )
    assumptions = [
        "Assumes local (windowed) stationarity of the mean and variance — both are re-estimated every period.",
        "Z-scores are undefined (NaN) wherever rolling standard deviation is zero or the window isn't yet full.",
    ]
    limitations = [
        "A high |Z-score| flags a statistically unusual move but is not itself a trading signal or forecast.",
        "Choice of window length materially changes results; no single window is 'correct' for all assets.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price"),
    ]
    params = [
        ParamSpec("security", "Security", "select", default=None, description="Security to analyze."),
        ParamSpec("window", "Rolling window (periods)", "int", default=20, min=5, max=252,
                   description="Trailing window for the rolling mean/stdev/Z-score."),
        ParamSpec("band_k", "Band width (std deviations)", "float", default=2.0, min=0.5, max=4.0,
                   description="Number of standard deviations for the price envelope."),
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
        security = params.get("security") or panel.columns[0]
        if security not in panel.columns:
            security = panel.columns[0]
        prices = panel[security].dropna()

        window = int(params.get("window", 20))
        k = float(params.get("band_k", 2.0))
        if len(prices) < window + 2:
            raise ValueError(f"Not enough observations ({len(prices)}) for a {window}-period rolling window.")

        roll_mean = prices.rolling(window).mean()
        roll_std = prices.rolling(window).std(ddof=1)
        upper = roll_mean + k * roll_std
        lower = roll_mean - k * roll_std

        returns = calc.simple_returns(prices)
        z = calc.rolling_zscore(returns, window)

        band_fig = go.Figure()
        band_fig.add_trace(go.Scatter(x=upper.index, y=upper.values, name=f"+{k}σ", line=dict(width=0.8, color="#94a3b8"), showlegend=True))
        band_fig.add_trace(go.Scatter(x=lower.index, y=lower.values, name=f"-{k}σ", line=dict(width=0.8, color="#94a3b8"),
                                       fill="tonexty", fillcolor="rgba(148,163,184,0.15)"))
        band_fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean.values, name=f"{window}-period Mean", line=dict(width=1, dash="dot")))
        band_fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name=f"{security} Price", line=dict(width=1.8)))
        apply_theme(band_fig, preset=params.get("theme", "professional"),
                    title=f"{security} — Price with {window}-period Rolling Bands (±{k}σ)", x_title="Date", y_title="Price")

        z_fig = go.Figure()
        z_fig.add_trace(go.Scatter(x=z.index, y=z.values, name="Rolling Z-Score", mode="lines", line=dict(width=1.5)))
        z_fig.add_hline(y=2, line=dict(color="#dc2626", dash="dash", width=1))
        z_fig.add_hline(y=-2, line=dict(color="#dc2626", dash="dash", width=1))
        z_fig.add_hline(y=0, line=dict(color="#94a3b8", width=1))
        apply_theme(z_fig, preset=params.get("theme", "professional"),
                    title=f"{security} — Rolling Z-Score of Returns (window={window})", x_title="Date", y_title="Z-Score", height=340)

        n_extreme = int((z.abs() > 2).sum())
        stats = {
            "Observations": len(prices),
            "Window": window,
            "Current Z-Score": round(float(z.dropna().iloc[-1]), 2) if z.dropna().shape[0] else None,
            "Periods with |Z| > 2": n_extreme,
            "% Periods with |Z| > 2": round(n_extreme / max(len(z.dropna()), 1) * 100, 1),
            "Current Price": round(float(prices.iloc[-1]), 4),
            "Current Rolling Mean": round(float(roll_mean.dropna().iloc[-1]), 4) if roll_mean.dropna().shape[0] else None,
        }

        rows = [
            {"date": str(d.date()), "price": float(prices.get(d, np.nan)),
             "rolling_mean": float(roll_mean.get(d, np.nan)), "rolling_std": float(roll_std.get(d, np.nan)),
             "upper_band": float(upper.get(d, np.nan)), "lower_band": float(lower.get(d, np.nan)),
             "return_zscore": float(z.get(d, np.nan)) if d in z.index else None}
            for d in prices.index
        ]

        return MethodResult(
            figure=self.fig_to_dict(band_fig),
            stats=stats,
            tables={"zscore_chart": self.fig_to_dict(z_fig)},
            series_csv_rows=rows,
            warnings=(["Using unadjusted Close prices."] if price_role_used == "close" else []),
        )
