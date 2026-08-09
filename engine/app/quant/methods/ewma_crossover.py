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


class EwmaCrossoverMethod(QuantMethod):
    id = "ewma_crossover"
    name = "EWMA Crossover Strategy"
    category = "Technical / Time-Series"
    difficulty = "intermediate"

    description = "Fast/slow exponentially-weighted moving average crossover signal, backtested against buy-and-hold."
    what_it_calculates = (
        "Two EWMAs of price (a fast, short-span average and a slow, long-span average). A long position is held "
        "when the fast EWMA is above the slow EWMA, flat/short otherwise. The resulting signal is backtested on "
        "the historical price path and compared to a passive buy-and-hold of the same security."
    )
    why_use_it = (
        "A simple, transparent trend-following signal. EWMA weights recent observations more heavily than a "
        "simple moving average (SMA), so it reacts faster to new information while still smoothing noise."
    )
    methodology = (
        "EWMA recursion: $E_t = \\alpha \\cdot P_t + (1-\\alpha) E_{t-1}$, with $\\alpha = 2/(span+1)$. "
        "Signal: long (1) when $E^{fast}_t > E^{slow}_t$, flat (0) otherwise (or short, -1, if enabled). "
        "Strategy return at $t$ uses the signal known at $t-1$ (no look-ahead): "
        "$r^{strategy}_t = signal_{t-1} \\times r_t - costs$. Transaction costs are charged on each signal change."
    )
    assumptions = [
        "Signals are lagged by one period before being applied to returns, so no future information leaks into "
        "the backtest (the trade decided at close of day t only affects the return realized on day t+1).",
        "Transaction costs and slippage are modeled as a flat basis-point charge applied on every position change.",
        "This is a historical backtest, not a live/forward test — past performance of the signal is not a "
        "guarantee of future results.",
    ]
    limitations = [
        "No borrowing costs, margin requirements, or capacity/liquidity constraints are modeled.",
        "A single-asset time-series signal; does not account for portfolio-level interactions.",
        "Parameter choices (fast/slow spans) were not optimized here — they are exactly what you specify, so "
        "results can be overstated if spans are tuned after seeing this same backtest (look-ahead via hindsight).",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price"),
    ]
    params = [
        ParamSpec("security", "Security", "select", default=None, description="Security to trade."),
        ParamSpec("fast_span", "Fast EWMA span (days)", "int", default=12, min=2, max=100,
                   description="Shorter span = more responsive fast average."),
        ParamSpec("slow_span", "Slow EWMA span (days)", "int", default=48, min=5, max=400,
                   description="Longer span = smoother slow average."),
        ParamSpec("allow_short", "Allow short positions", "bool", default=False,
                   description="If off, the strategy is long/flat only."),
        ParamSpec("cost_bps", "Transaction cost (bps per position change)", "float", default=5.0, min=0, max=200,
                   description="Round-trip-agnostic flat cost charged on every signal change, in basis points."),
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
        warnings: list[str] = []

        prices, security = resolve_security_series(panel, params.get("security"))

        fast_span = int(params.get("fast_span", 12))
        slow_span = int(params.get("slow_span", 48))
        if fast_span >= slow_span:
            raise ValueError(f"Fast span ({fast_span}) must be smaller than slow span ({slow_span}).")
        if len(prices) < slow_span + 5:
            raise ValueError(
                f"Not enough observations ({len(prices)}) for a slow span of {slow_span}. "
                f"Need at least {slow_span + 5}."
            )

        allow_short = bool(params.get("allow_short", False))
        cost_bps = float(params.get("cost_bps", 5.0)) / 10000.0

        fast = calc.ewma(prices, fast_span)
        slow = calc.ewma(prices, slow_span)
        raw_signal = np.where(fast > slow, 1, (-1 if allow_short else 0))
        signal = pd.Series(raw_signal, index=prices.index)

        returns = calc.simple_returns(prices)
        lagged_signal = signal.shift(1).reindex(returns.index).fillna(0)
        position_changes = lagged_signal.diff().abs().fillna(lagged_signal.abs())
        costs = position_changes * cost_bps
        strategy_returns = lagged_signal * returns - costs

        strat_equity = calc.equity_curve(strategy_returns)
        bh_equity = calc.equity_curve(returns)

        crossings = signal.diff().fillna(0) != 0
        entries = signal[(crossings) & (signal == 1)]
        exits = signal[(crossings) & (signal != 1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name=f"{security} Price",
                                  mode="lines", line=dict(width=1.2, color="#94a3b8")))
        fig.add_trace(go.Scatter(x=fast.index, y=fast.values, name=f"Fast EWMA ({fast_span})", mode="lines"))
        fig.add_trace(go.Scatter(x=slow.index, y=slow.values, name=f"Slow EWMA ({slow_span})", mode="lines"))
        if len(entries):
            fig.add_trace(go.Scatter(x=entries.index, y=prices.reindex(entries.index), mode="markers",
                                      name="Long entry", marker=dict(symbol="triangle-up", size=10, color="#059669")))
        if len(exits):
            fig.add_trace(go.Scatter(x=exits.index, y=prices.reindex(exits.index), mode="markers",
                                      name="Exit / short", marker=dict(symbol="triangle-down", size=10, color="#dc2626")))
        apply_theme(fig, preset=params.get("theme", "professional"),
                    title=f"{security} — EWMA({fast_span}/{slow_span}) Crossover", x_title="Date", y_title="Price")

        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=strat_equity.index, y=strat_equity.values, name="EWMA Strategy",
                                     mode="lines", line=dict(width=2)))
        eq_fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity.values, name="Buy & Hold",
                                     mode="lines", line=dict(width=1.5, dash="dot")))
        apply_theme(eq_fig, preset=params.get("theme", "professional"),
                    title="Strategy vs. Buy-and-Hold — Growth of $1", x_title="Date", y_title="Portfolio Value ($)")

        dd = calc.drawdown_series(strat_equity)
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, name="Strategy Drawdown", mode="lines",
                                     fill="tozeroy", line=dict(width=1, color="#dc2626")))
        apply_theme(dd_fig, preset=params.get("theme", "professional"), title="Strategy Drawdown", y_title="Drawdown (%)",
                    height=280)

        n = len(strategy_returns)
        strat_total = float(strat_equity.iloc[-1] - 1) if n else float("nan")
        bh_total = float(bh_equity.iloc[-1] - 1) if len(bh_equity) else float("nan")
        strat_cagr = calc.annualize_return(strat_total, n)
        bh_cagr = calc.annualize_return(bh_total, len(returns))
        strat_max_dd, _, _ = calc.max_drawdown(strat_equity)
        n_trades = int(position_changes[position_changes > 0].count())

        stats = {
            "Strategy Total Return (%)": round(strat_total * 100, 2),
            "Buy & Hold Total Return (%)": round(bh_total * 100, 2),
            "Strategy CAGR (%)": round(strat_cagr * 100, 2),
            "Buy & Hold CAGR (%)": round(bh_cagr * 100, 2),
            "Strategy Annualized Volatility (%)": round(calc.annualize_vol(strategy_returns) * 100, 2),
            "Strategy Sharpe Ratio": round(calc.sharpe_ratio(strategy_returns), 2),
            "Strategy Max Drawdown (%)": round(strat_max_dd * 100, 2),
            "Number of Position Changes": n_trades,
            "Time in Market (%)": round(float((lagged_signal != 0).mean()) * 100, 1),
        }

        rows = [
            {"date": str(d.date()), "price": float(prices.get(d, np.nan)),
             "fast_ewma": float(fast.get(d, np.nan)), "slow_ewma": float(slow.get(d, np.nan)),
             "signal": int(signal.get(d, 0)),
             "strategy_return": float(strategy_returns.get(d, np.nan)) if d in strategy_returns.index else None,
             "strategy_equity": float(strat_equity.get(d, np.nan)) if d in strat_equity.index else None,
             "buy_hold_equity": float(bh_equity.get(d, np.nan)) if d in bh_equity.index else None}
            for d in prices.index
        ]

        if price_role_used == "close":
            warnings.append("Using unadjusted Close prices — signal/backtest may be distorted around dividends/splits.")

        return MethodResult(
            figure=self.fig_to_dict(fig),
            stats=stats,
            tables={"equity_curve": self.fig_to_dict(eq_fig), "drawdown": self.fig_to_dict(dd_fig)},
            series_csv_rows=rows,
            warnings=warnings,
        )
