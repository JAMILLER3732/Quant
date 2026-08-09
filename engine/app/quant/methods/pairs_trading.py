from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from app.data.reshape import price_panel
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme


class PairsTradingMethod(QuantMethod):
    id = "pairs_trading"
    name = "Pairs Trading & Cointegration"
    category = "Pairs Trading / Mean Reversion"
    difficulty = "advanced"

    description = "Engle-Granger cointegration test, hedge ratio, spread Z-score, half-life of mean reversion, and a backtested pairs-trading signal for two securities."
    what_it_calculates = (
        "Whether two price series are statistically cointegrated (Engle-Granger test), the OLS hedge ratio "
        "between them, the resulting spread and its rolling Z-score, the spread's mean-reversion half-life, "
        "and a backtested long/short pairs-trading signal based on Z-score thresholds."
    )
    why_use_it = (
        "The standard statistical-arbitrage framework: trade the spread between two related securities when it "
        "diverges from its historical relationship, betting on reversion — but only if that relationship is "
        "actually statistically supported (cointegration), not just visually appealing."
    )
    methodology = (
        "Hedge ratio $\\beta$ from static OLS: $P_A = \\alpha + \\beta P_B + \\epsilon$. Spread: "
        "$S_t = P_{A,t} - \\beta P_{B,t}$. Engle-Granger test (statsmodels `coint`) checks whether $S_t$ is "
        "stationary (i.e. $A$ and $B$ are cointegrated) — a p-value below 0.05 is standard evidence of "
        "cointegration. Half-life of mean reversion comes from fitting "
        "$\\Delta S_t = \\theta (S_{t-1} - \\bar S) + \\epsilon_t$ by OLS and computing "
        "$\\text{half-life} = -\\ln(2)/\\theta$ (only meaningful when $\\theta < 0$). Rolling Z-score of the "
        "spread drives entry ($|z| \\ge z_{entry}$) and exit ($|z| \\le z_{exit}$) signals, backtested with a "
        "one-period signal lag."
    )
    assumptions = [
        "The hedge ratio is estimated once via OLS over the full sample (not rolling) and held fixed through "
        "the backtest — a static, in-sample hedge ratio.",
        "Trading the spread means being long one leg and short the other in the ratio 1 : β — this requires "
        "shorting to be feasible for security B.",
        "Signals are lagged one period before being applied to returns — no look-ahead bias.",
    ]
    limitations = [
        "A cointegration p-value below 0.05 is evidence, not proof, of a stable statistical relationship — "
        "relationships can and do break down (\"cointegration breakdown\" risk).",
        "The static hedge ratio does not adapt if the true relationship between the securities drifts over "
        "time; a rolling hedge ratio would adapt but is not computed here.",
        "No borrowing costs for the short leg are modeled.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price", min_series=2,
                       note="Need exactly 2 securities to form a pair."),
    ]
    params = [
        ParamSpec("security_a", "Security A", "select", default=None, description="First leg of the pair."),
        ParamSpec("security_b", "Security B", "select", default=None, description="Second leg of the pair."),
        ParamSpec("window", "Rolling Z-score window (days)", "int", default=20, min=5, max=252,
                   description="Window for the rolling spread mean/stdev."),
        ParamSpec("entry_z", "Entry Z-score threshold", "float", default=2.0, min=0.5, max=5.0,
                   description="Enter the spread trade when |Z| exceeds this."),
        ParamSpec("exit_z", "Exit Z-score threshold", "float", default=0.5, min=0.0, max=3.0,
                   description="Exit when |Z| falls back below this."),
        ParamSpec("cost_bps", "Transaction cost (bps per entry/exit)", "float", default=5.0, min=0, max=200,
                   description="Flat cost on every position change, charged on both legs combined."),
    ]

    def check_requirements(self, role_map, df):
        base = super().check_requirements(role_map, df)
        try:
            panel, _ = price_panel(df, role_map)
            if panel.shape[1] < 2:
                base.satisfied = False
                base.missing.append(f"Found only {panel.shape[1]} security — need exactly 2 for a pair.")
        except Exception:
            base.satisfied = False
            base.missing.append("Map at least 2 securities' Close/Adjusted Close prices.")
        return base

    def calculate(self, df: pd.DataFrame, role_map: dict[str, str], params: dict[str, Any]) -> MethodResult:
        panel, price_role_used = price_panel(df, role_map)
        columns = list(panel.columns)
        sec_a = params.get("security_a") or columns[0]
        sec_b = params.get("security_b") or (columns[1] if len(columns) > 1 else columns[0])
        if sec_a not in columns:
            sec_a = columns[0]
        if sec_b not in columns or sec_b == sec_a:
            sec_b = next((c for c in columns if c != sec_a), columns[0])
        if sec_a == sec_b:
            raise ValueError("Security A and Security B must be different securities.")

        prices = panel[[sec_a, sec_b]].dropna()
        if len(prices) < 60:
            raise ValueError(f"Only {len(prices)} overlapping observations for this pair — need at least 60.")

        p_a, p_b = prices[sec_a], prices[sec_b]
        hedge_model = sm.OLS(p_a.values, sm.add_constant(p_b.values)).fit()
        hedge_ratio = float(hedge_model.params[1])
        spread = p_a - hedge_ratio * p_b

        coint_stat, coint_pvalue, _ = coint(p_a.values, p_b.values)

        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        aligned_idx = spread_lag.index.intersection(spread_diff.index)
        hl_model = sm.OLS(spread_diff.loc[aligned_idx].values,
                           sm.add_constant(spread_lag.loc[aligned_idx].values - spread.mean())).fit()
        theta = float(hl_model.params[1])
        half_life = -np.log(2) / theta if theta < 0 else float("nan")

        window = int(params.get("window", 20))
        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std(ddof=1)
        z = (spread - roll_mean) / roll_std

        entry_z = float(params.get("entry_z", 2.0))
        exit_z = float(params.get("exit_z", 0.5))
        if exit_z >= entry_z:
            raise ValueError(f"Exit threshold ({exit_z}) must be smaller than entry threshold ({entry_z}).")
        cost = float(params.get("cost_bps", 5.0)) / 10000.0

        position, positions = 0, []
        for date in spread.index:
            zt = z.get(date, np.nan)
            if position == 0 and not np.isnan(zt):
                if zt >= entry_z:
                    position = -1  # spread too high: short A, long B
                elif zt <= -entry_z:
                    position = 1   # spread too low: long A, short B
            elif position != 0 and not np.isnan(zt) and abs(zt) <= exit_z:
                position = 0
            positions.append(position)
        signal = pd.Series(positions, index=spread.index)

        spread_returns = spread.diff() / p_b.shift(1).abs()  # normalize by leg B notional as a simple P&L proxy
        lagged_signal = signal.shift(1).reindex(spread_returns.index).fillna(0)
        changes = lagged_signal.diff().abs().fillna(lagged_signal.abs())
        strat_returns = (lagged_signal * spread_returns - changes * cost).dropna()
        strat_equity = (1 + strat_returns).cumprod()

        spread_fig = go.Figure()
        spread_fig.add_trace(go.Scatter(x=spread.index, y=spread.values, name="Spread", line=dict(width=1.5)))
        spread_fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean.values, name=f"{window}-day Mean", line=dict(width=1, dash="dot")))
        apply_theme(spread_fig, preset=params.get("theme", "professional"),
                    title=f"{sec_a} vs {sec_b} — Spread (hedge ratio β={hedge_ratio:.3f})",
                    subtitle=f"Engle-Granger p-value={coint_pvalue:.4f} ({'cointegrated' if coint_pvalue < 0.05 else 'not significantly cointegrated'} at 5%)",
                    x_title="Date", y_title="Spread")

        z_fig = go.Figure()
        z_fig.add_trace(go.Scatter(x=z.index, y=z.values, name="Spread Z-score", line=dict(width=1.5)))
        z_fig.add_hline(y=entry_z, line=dict(color="#dc2626", dash="dash", width=1))
        z_fig.add_hline(y=-entry_z, line=dict(color="#dc2626", dash="dash", width=1))
        z_fig.add_hline(y=0, line=dict(color="#94a3b8", width=1))
        apply_theme(z_fig, preset=params.get("theme", "professional"), title="Spread Z-Score", x_title="Date", y_title="Z-score", height=320)

        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=strat_equity.index, y=strat_equity.values, name="Pairs Strategy", line=dict(width=2)))
        apply_theme(eq_fig, preset=params.get("theme", "professional"), title="Pairs Strategy — Growth of $1 (notional-normalized)",
                    x_title="Date", y_title="Value", height=320)

        stats = {
            "Hedge Ratio (β)": round(hedge_ratio, 4),
            "Engle-Granger Test Statistic": round(float(coint_stat), 3),
            "Engle-Granger p-value": round(float(coint_pvalue), 4),
            "Cointegrated at 5%": bool(coint_pvalue < 0.05),
            "Half-Life of Mean Reversion (days)": round(float(half_life), 1) if half_life == half_life else None,
            "Current Spread Z-score": round(float(z.dropna().iloc[-1]), 2) if z.dropna().shape[0] else None,
            "Number of Round-Trip Trades": int((lagged_signal.diff().fillna(0) != 0).sum() // 2),
            "Strategy Total Return (%)": round(float(strat_equity.iloc[-1] - 1) * 100, 2) if len(strat_equity) else None,
        }

        warnings = []
        if price_role_used == "close":
            warnings.append("Using unadjusted Close prices for at least one leg.")
        if coint_pvalue >= 0.05:
            warnings.append(
                f"Engle-Granger p-value is {coint_pvalue:.3f} (≥0.05) — this pair does NOT show statistically "
                "significant cointegration over this sample. Mean-reversion trading on this spread is not "
                "statistically supported by this test."
            )

        rows = [
            {"date": str(d.date()), "spread": float(spread.get(d, np.nan)), "z_score": float(z.get(d, np.nan)) if d in z.index else None,
             "position": int(signal.get(d, 0))}
            for d in spread.index
        ]

        return MethodResult(
            figure=self.fig_to_dict(spread_fig),
            stats=stats,
            tables={"zscore_chart": self.fig_to_dict(z_fig), "strategy_equity": self.fig_to_dict(eq_fig)},
            series_csv_rows=rows,
            warnings=warnings,
        )
