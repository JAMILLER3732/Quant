from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.data.reshape import price_panel
from app.quant import calc
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme

REGIME_COLORS = {"Low Vol": "#059669", "Medium Vol": "#f59e0b", "High Vol": "#dc2626"}


class RegimeAnalysisMethod(QuantMethod):
    id = "regime_analysis"
    name = "Volatility & Trend Regime Analysis"
    category = "Regime Analysis"
    difficulty = "intermediate"

    description = "Classifies each period into a Low/Medium/High volatility regime (by rolling-volatility terciles) and a Bull/Bear trend regime (price vs. its long moving average), with per-regime statistics."
    what_it_calculates = (
        "Two independent, transparent regime classifications: (1) a volatility regime — Low/Medium/High, "
        "based on which tercile of the historical rolling-volatility distribution each period falls into — and "
        "(2) a trend regime — Bull/Bear, based on whether price is above or below its long-window moving "
        "average. Reports mean return, volatility, and time-in-regime for each."
    )
    why_use_it = (
        "Returns and risk are not stationary through time — this makes that visible and quantifies how "
        "performance differs across volatility and trend environments, without requiring an unstable "
        "black-box classifier."
    )
    methodology = (
        "Volatility regime: rolling annualized volatility $\\sigma_t^{(w)}$ is computed over window $w$, then "
        "each observation is labeled Low/Medium/High by which historical-sample tercile (33rd/67th percentile "
        "of $\\sigma_t^{(w)}$ over the full series) it falls into. Trend regime: Bull if $P_t > SMA_t^{(n)}$, "
        "Bear otherwise, for moving-average window $n$. Both are simple, deterministic, fully reproducible "
        "classifications — not a fitted Hidden Markov Model or unsupervised clustering, which would add model "
        "risk without more data to validate against."
    )
    assumptions = [
        "Volatility terciles are computed over the same sample being displayed (in-sample thresholds) — the "
        "labels describe historical regimes, not real-time forward classification.",
        "Trend regime uses a single moving-average crossover convention; other definitions (e.g. multiple "
        "MAs, drawdown-based) would classify periods differently.",
    ]
    limitations = [
        "This is a descriptive, rule-based classification, not a statistical latent-regime model (e.g. Hidden "
        "Markov Model) — it will not detect regime structure the chosen rules don't already encode.",
        "Regime boundaries are somewhat arbitrary (tercile cutoffs, MA window length); different choices can "
        "shift the classification meaningfully.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price"),
    ]
    params = [
        ParamSpec("security", "Security", "select", default=None, description="Security to analyze."),
        ParamSpec("vol_window", "Volatility window (days)", "int", default=21, min=5, max=252,
                   description="Rolling window for the volatility regime classification."),
        ParamSpec("trend_window", "Trend moving-average window (days)", "int", default=200, min=20, max=400,
                   description="Window for the bull/bear moving-average."),
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

        vol_window = int(params.get("vol_window", 21))
        trend_window = int(params.get("trend_window", 200))
        if len(prices) < max(vol_window, trend_window) + 10:
            raise ValueError(f"Not enough observations ({len(prices)}) for the requested windows (vol={vol_window}, trend={trend_window}).")

        returns = calc.simple_returns(prices)
        roll_vol = returns.rolling(vol_window).std(ddof=1) * np.sqrt(calc.TRADING_DAYS_PER_YEAR)
        roll_vol_clean = roll_vol.dropna()
        q33, q67 = roll_vol_clean.quantile([1 / 3, 2 / 3])

        def vol_label(v):
            if np.isnan(v):
                return None
            if v <= q33:
                return "Low Vol"
            if v <= q67:
                return "Medium Vol"
            return "High Vol"

        vol_regime = roll_vol.apply(vol_label)

        sma = prices.rolling(trend_window).mean()
        trend_regime = pd.Series(np.where(prices > sma, "Bull", "Bear"), index=prices.index)
        trend_regime[sma.isna()] = None

        price_fig = go.Figure()
        for label, color in REGIME_COLORS.items():
            # vol_regime is indexed on returns.index (one shorter than prices.index,
            # since returns drop the first observation) — match by date via .loc,
            # not a positional boolean mask, to avoid a length mismatch against prices.
            matched_dates = vol_regime.index[vol_regime == label]
            if len(matched_dates):
                price_fig.add_trace(go.Scatter(
                    x=matched_dates, y=prices.loc[matched_dates].values, mode="markers",
                    marker=dict(size=3, color=color), name=label,
                ))
        price_fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode="lines",
                                        line=dict(width=0.8, color="#94a3b8"), name="Price", showlegend=False, hoverinfo="skip"))
        apply_theme(price_fig, preset=params.get("theme", "professional"),
                    title=f"{security} — Price Colored by Volatility Regime", x_title="Date", y_title="Price")

        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name="Price", line=dict(width=1.3)))
        trend_fig.add_trace(go.Scatter(x=sma.index, y=sma.values, name=f"{trend_window}-day SMA", line=dict(width=1.3, dash="dot")))
        bear_mask = trend_regime == "Bear"
        if bear_mask.any():
            trend_fig.add_trace(go.Scatter(x=prices.index[bear_mask], y=prices.values[bear_mask], mode="markers",
                                            marker=dict(size=3, color="#dc2626"), name="Bear regime"))
        apply_theme(trend_fig, preset=params.get("theme", "professional"),
                    title=f"{security} — Trend Regime (Bull/Bear vs {trend_window}-day SMA)", x_title="Date", y_title="Price", height=380)

        regime_df = pd.DataFrame({"return": returns, "vol_regime": vol_regime.reindex(returns.index),
                                   "trend_regime": trend_regime.reindex(returns.index)}).dropna(subset=["return"])

        vol_stats_table = []
        for label in ("Low Vol", "Medium Vol", "High Vol"):
            sub = regime_df[regime_df["vol_regime"] == label]["return"]
            if len(sub):
                vol_stats_table.append({
                    "Regime": label, "Observations": len(sub), "% of Time": round(len(sub) / len(regime_df) * 100, 1),
                    "Mean Daily Return (%)": round(float(sub.mean()) * 100, 4),
                    "Annualized Volatility (%)": round(float(sub.std(ddof=1)) * np.sqrt(calc.TRADING_DAYS_PER_YEAR) * 100, 2),
                })

        trend_stats_table = []
        for label in ("Bull", "Bear"):
            sub = regime_df[regime_df["trend_regime"] == label]["return"]
            if len(sub):
                trend_stats_table.append({
                    "Regime": label, "Observations": len(sub), "% of Time": round(len(sub) / len(regime_df) * 100, 1),
                    "Mean Daily Return (%)": round(float(sub.mean()) * 100, 4),
                    "Annualized Return (%)": round(float(sub.mean()) * calc.TRADING_DAYS_PER_YEAR * 100, 2),
                })

        stats = {
            "Current Volatility Regime": vol_regime.dropna().iloc[-1] if vol_regime.dropna().shape[0] else None,
            "Current Trend Regime": trend_regime.dropna().iloc[-1] if trend_regime.dropna().shape[0] else None,
            "Low/Medium Vol Threshold (annualized, %)": round(float(q33) * 100, 2),
            "Medium/High Vol Threshold (annualized, %)": round(float(q67) * 100, 2),
            "Observations": len(regime_df),
        }

        rows = [
            {"date": str(d.date()), "price": float(prices.get(d, np.nan)), "volatility_regime": vol_regime.get(d),
             "trend_regime": trend_regime.get(d), "rolling_volatility_annualized": float(roll_vol.get(d, np.nan))}
            for d in prices.index
        ]

        return MethodResult(
            figure=self.fig_to_dict(price_fig),
            stats=stats,
            tables={"trend_regime_chart": self.fig_to_dict(trend_fig), "volatility_regime_stats": vol_stats_table,
                    "trend_regime_stats": trend_stats_table},
            series_csv_rows=rows,
            warnings=(["Using unadjusted Close prices."] if price_role_used == "close" else []),
        )
