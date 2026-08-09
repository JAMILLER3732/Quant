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


class MeanReversionBacktestMethod(QuantMethod):
    id = "mean_reversion_backtest"
    name = "Mean-Reversion Z-Score Backtest"
    category = "Backtesting"
    difficulty = "advanced"

    description = "Rule-based long/short backtest: enter when price Z-score crosses an extreme threshold, exit on reversion, stop-loss, or take-profit — with transaction costs."
    what_it_calculates = (
        "A full historical backtest of a mean-reversion trading rule: go long when price is statistically far "
        "below its rolling mean, short when far above, exit as it reverts toward the mean (or hits a stop-loss/"
        "take-profit), and reports the resulting equity curve, drawdown, trade list, and performance stats "
        "versus buy-and-hold."
    )
    why_use_it = (
        "Demonstrates a complete, leak-free backtesting loop with explicit entry/exit/risk rules — the pattern "
        "this platform's more advanced strategy tools build on."
    )
    methodology = (
        "Rolling Z-score: $z_t = (P_t - \\mu_t^{(w)}) / \\sigma_t^{(w)}$ over trailing window $w$. Entry: go long "
        "(short) when $z_t \\le -z_{entry}$ ($z_t \\ge z_{entry}$) and flat. Exit: close the position when "
        "$|z_t| \\le z_{exit}$, or when unrealized P&L breaches the stop-loss/take-profit bounds, whichever "
        "comes first. The signal decided using information available at close of day $t$ is applied to the "
        "return realized on day $t+1$ (one-period lag) — no future information is used. Transaction costs are "
        "charged (in basis points) on every entry and exit."
    )
    assumptions = [
        "Signals are lagged one period before being applied to returns — no look-ahead bias.",
        "Position sizing is binary (fully in or flat/short), not scaled by conviction.",
        "Stop-loss/take-profit are evaluated once per day against the closing price, not intraday.",
        "This is a historical backtest on a single security, not a forward/out-of-sample test — see the "
        "Limitations below.",
    ]
    limitations = [
        "A backtest measures how a fixed rule would have performed historically; it is not a guarantee of "
        "future performance, especially if the parameters were chosen by looking at this same backtest "
        "(in-sample overfitting).",
        "No slippage beyond the flat transaction-cost assumption, no borrowing cost for short positions, and no "
        "capacity/liquidity constraints are modeled.",
        "Mean reversion can fail for extended periods if the underlying trend changes — the stop-loss limits "
        "but does not eliminate this risk.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price"),
    ]
    params = [
        ParamSpec("security", "Security", "select", default=None, description="Security to trade."),
        ParamSpec("window", "Rolling window (days)", "int", default=20, min=5, max=252,
                   description="Window for the rolling mean/stdev used to compute the Z-score."),
        ParamSpec("entry_z", "Entry Z-score threshold", "float", default=2.0, min=0.5, max=5.0,
                   description="Enter a position when |Z| exceeds this value."),
        ParamSpec("exit_z", "Exit Z-score threshold", "float", default=0.5, min=0.0, max=3.0,
                   description="Exit when |Z| falls back below this value."),
        ParamSpec("allow_short", "Allow short positions", "bool", default=True,
                   description="If off, only long entries (on oversold signals) are taken."),
        ParamSpec("stop_loss_pct", "Stop-loss (%)", "float", default=10.0, min=1, max=50,
                   description="Force-exit if unrealized loss on the position exceeds this."),
        ParamSpec("take_profit_pct", "Take-profit (%)", "float", default=15.0, min=1, max=100,
                   description="Force-exit if unrealized gain on the position exceeds this."),
        ParamSpec("cost_bps", "Transaction cost (bps per entry/exit)", "float", default=5.0, min=0, max=200,
                   description="Flat cost charged on every position change."),
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

        window = int(params.get("window", 20))
        entry_z = float(params.get("entry_z", 2.0))
        exit_z = float(params.get("exit_z", 0.5))
        if exit_z >= entry_z:
            raise ValueError(f"Exit threshold ({exit_z}) must be smaller than entry threshold ({entry_z}).")
        allow_short = bool(params.get("allow_short", True))
        stop_loss = float(params.get("stop_loss_pct", 10.0)) / 100.0
        take_profit = float(params.get("take_profit_pct", 15.0)) / 100.0
        cost = float(params.get("cost_bps", 5.0)) / 10000.0

        if len(prices) < window + 10:
            raise ValueError(f"Not enough observations ({len(prices)}) for a {window}-day rolling window.")

        z = calc.rolling_zscore(prices, window)

        # Sequential state machine: entry/exit/stop-loss/take-profit are inherently
        # path-dependent, so this loop runs once over the series (O(n)), not per-simulation.
        position = 0
        entry_price = None
        positions = []
        trade_log = []
        for date, price in prices.items():
            z_t = z.get(date, np.nan)
            if position == 0:
                if not np.isnan(z_t):
                    if z_t <= -entry_z:
                        position = 1
                        entry_price = price
                        trade_log.append({"date": str(date.date()), "action": "enter_long", "price": float(price), "z": float(z_t)})
                    elif allow_short and z_t >= entry_z:
                        position = -1
                        entry_price = price
                        trade_log.append({"date": str(date.date()), "action": "enter_short", "price": float(price), "z": float(z_t)})
            else:
                unrealized = position * (price / entry_price - 1.0)
                exit_reason = None
                if not np.isnan(z_t) and abs(z_t) <= exit_z:
                    exit_reason = "reversion"
                elif unrealized <= -stop_loss:
                    exit_reason = "stop_loss"
                elif unrealized >= take_profit:
                    exit_reason = "take_profit"
                if exit_reason:
                    trade_log.append({"date": str(date.date()), "action": f"exit_{exit_reason}", "price": float(price),
                                       "z": float(z_t) if not np.isnan(z_t) else None, "pnl_pct": round(unrealized * 100, 2)})
                    position = 0
                    entry_price = None
            positions.append(position)

        signal = pd.Series(positions, index=prices.index)
        returns = calc.simple_returns(prices)
        lagged_signal = signal.shift(1).reindex(returns.index).fillna(0)
        changes = lagged_signal.diff().abs().fillna(lagged_signal.abs())
        costs = changes * cost
        strategy_returns = lagged_signal * returns - costs
        strat_equity = calc.equity_curve(strategy_returns)
        bh_equity = calc.equity_curve(returns)
        dd = calc.drawdown_series(strat_equity)

        n_trades = sum(1 for t in trade_log if t["action"].startswith("enter"))
        wins = [t for t in trade_log if t["action"].startswith("exit") and t.get("pnl_pct", 0) > 0]
        exits = [t for t in trade_log if t["action"].startswith("exit")]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode="lines", name=security, line=dict(width=1.3, color="#94a3b8")))
        entries_long = [t for t in trade_log if t["action"] == "enter_long"]
        entries_short = [t for t in trade_log if t["action"] == "enter_short"]
        if entries_long:
            fig.add_trace(go.Scatter(x=[t["date"] for t in entries_long], y=[t["price"] for t in entries_long],
                                      mode="markers", name="Long entry", marker=dict(symbol="triangle-up", size=10, color="#059669")))
        if entries_short:
            fig.add_trace(go.Scatter(x=[t["date"] for t in entries_short], y=[t["price"] for t in entries_short],
                                      mode="markers", name="Short entry", marker=dict(symbol="triangle-down", size=10, color="#dc2626")))
        if exits:
            fig.add_trace(go.Scatter(x=[t["date"] for t in exits], y=[t["price"] for t in exits],
                                      mode="markers", name="Exit", marker=dict(symbol="x", size=8, color="#f59e0b")))
        apply_theme(fig, preset=params.get("theme", "professional"),
                    title=f"{security} — Mean-Reversion Backtest (window={window}, entry|z|≥{entry_z}, exit|z|≤{exit_z})",
                    x_title="Date", y_title="Price")

        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=strat_equity.index, y=strat_equity.values, name="Strategy", line=dict(width=2)))
        eq_fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity.values, name="Buy & Hold", line=dict(width=1.5, dash="dot")))
        apply_theme(eq_fig, preset=params.get("theme", "professional"), title="Strategy vs. Buy-and-Hold — Growth of $1",
                    x_title="Date", y_title="Portfolio Value ($)")

        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, fill="tozeroy", line=dict(width=1, color="#dc2626"), name="Drawdown"))
        apply_theme(dd_fig, preset=params.get("theme", "professional"), title="Strategy Drawdown", y_title="Drawdown (%)", height=280)

        strat_total = float(strat_equity.iloc[-1] - 1) if len(strat_equity) else float("nan")
        bh_total = float(bh_equity.iloc[-1] - 1) if len(bh_equity) else float("nan")
        max_dd, _, _ = calc.max_drawdown(strat_equity)

        stats = {
            "Strategy Total Return (%)": round(strat_total * 100, 2),
            "Buy & Hold Total Return (%)": round(bh_total * 100, 2),
            "Strategy CAGR (%)": round(calc.annualize_return(strat_total, len(strategy_returns)) * 100, 2),
            "Strategy Sharpe Ratio": round(calc.sharpe_ratio(strategy_returns), 2),
            "Strategy Max Drawdown (%)": round(max_dd * 100, 2),
            "Number of Trades": n_trades,
            "Win Rate (%)": round(len(wins) / len(exits) * 100, 1) if exits else None,
            "Time in Market (%)": round(float((lagged_signal != 0).mean()) * 100, 1),
        }

        warnings = []
        if price_role_used == "close":
            warnings.append("Using unadjusted Close prices — signal/backtest may be distorted around dividends/splits.")
        if n_trades < 5:
            warnings.append(f"Only {n_trades} trade(s) were generated — performance statistics are not statistically reliable with so few trades.")

        return MethodResult(
            figure=self.fig_to_dict(fig),
            stats=stats,
            tables={"equity_curve": self.fig_to_dict(eq_fig), "drawdown": self.fig_to_dict(dd_fig), "trade_log": trade_log},
            series_csv_rows=[
                {"date": str(d.date()), "price": float(prices.get(d, np.nan)), "z_score": float(z.get(d, np.nan)) if d in z.index else None,
                 "position": int(signal.get(d, 0)), "strategy_return": float(strategy_returns.get(d, np.nan)) if d in strategy_returns.index else None,
                 "strategy_equity": float(strat_equity.get(d, np.nan)) if d in strat_equity.index else None}
                for d in prices.index
            ],
            warnings=warnings,
        )
