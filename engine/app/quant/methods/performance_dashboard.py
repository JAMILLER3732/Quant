from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.data.reshape import price_panel
from app.quant import calc
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme


class PerformanceDashboardMethod(QuantMethod):
    id = "performance_dashboard"
    name = "Performance & Risk Dashboard"
    category = "Risk Analytics"
    difficulty = "beginner"

    description = "Sharpe, Sortino, Calmar, max drawdown, VaR/CVaR and equity curves — compare every mapped security side by side."
    what_it_calculates = (
        "For every security with a mapped price series: total return, CAGR, annualized volatility, Sharpe ratio, "
        "Sortino ratio, Calmar ratio, maximum drawdown, historical VaR and Expected Shortfall — ranked in a table, "
        "plus equity-curve and risk/return scatter charts."
    )
    why_use_it = (
        "The standard first pass for comparing multiple securities or strategies on a like-for-like basis before "
        "deeper risk or optimization work."
    )
    methodology = (
        "Sharpe: $\\frac{E[r-r_f]}{\\sigma(r-r_f)}\\sqrt{252}$. Sortino: same numerator, denominator is downside "
        "deviation below the risk-free rate. Calmar: $CAGR / |MaxDD|$. Historical VaR/CVaR at the chosen confidence "
        "level are the empirical quantile / tail-mean of the historical return distribution (no distributional "
        "assumption)."
    )
    assumptions = [
        "252 trading days/year annualization.",
        "Risk-free rate is a single annualized constant, applied uniformly across the sample.",
        "Historical VaR/CVaR is unconditional (equal weight to every historical observation) — it assumes the "
        "future resembles the historical sample.",
    ]
    limitations = [
        "Ranking by Sharpe/Sortino can be unstable over short samples or in non-normal return regimes.",
        "Historical VaR/CVaR will understate risk if the sample window did not include a tail event.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price"),
    ]
    params = [
        ParamSpec("risk_free_rate", "Risk-free rate (annualized, %)", "float", default=0.0, min=-5, max=20,
                   description="Used for Sharpe/Sortino excess-return calculations."),
        ParamSpec("var_confidence", "VaR / ES confidence level (%)", "float", default=95, min=80, max=99.9,
                   description="Confidence level for historical VaR and Expected Shortfall."),
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
        rf = float(params.get("risk_free_rate", 0.0)) / 100.0
        confidence = float(params.get("var_confidence", 95.0)) / 100.0

        rows = []
        equity_by_security: dict[str, pd.Series] = {}
        scatter_x, scatter_y, scatter_labels = [], [], []

        for security in panel.columns:
            prices = panel[security].dropna()
            if len(prices) < 10:
                continue
            returns = calc.simple_returns(prices)
            equity = calc.equity_curve(returns)
            total_return = float(equity.iloc[-1] - 1)
            cagr = calc.annualize_return(total_return, len(returns))
            vol = calc.annualize_vol(returns)
            sharpe = calc.sharpe_ratio(returns, rf)
            sortino = calc.sortino_ratio(returns, rf)
            max_dd, _, _ = calc.max_drawdown(equity)
            calmar = calc.calmar_ratio(cagr, max_dd)
            var = calc.historical_var(returns, confidence)
            es = calc.expected_shortfall(returns, confidence)

            rows.append({
                "Security": security, "CAGR (%)": round(cagr * 100, 2),
                "Volatility (%)": round(vol * 100, 2), "Sharpe": round(sharpe, 2),
                "Sortino": round(sortino, 2), "Calmar": round(calmar, 2),
                "Max Drawdown (%)": round(max_dd * 100, 2),
                f"VaR {int(confidence*100)}% (%)": round(var * 100, 2),
                f"CVaR {int(confidence*100)}% (%)": round(es * 100, 2),
                "Observations": len(returns),
            })

            equity_by_security[security] = equity
            scatter_x.append(vol * 100)
            scatter_y.append(cagr * 100)
            scatter_labels.append(security)

        if not rows:
            raise ValueError("None of the mapped securities have enough observations (need at least 10) to compute performance statistics.")

        # With more than a handful of securities, a legend entry (and a static
        # text label on the scatter) per line becomes an unreadable overlapping
        # wall of text — and with only 8 palette colors, most lines would share
        # a color anyway, making the legend actively misleading past 8 series.
        # Past MAX_HIGHLIGHTED securities, draw every line but only name and
        # color the best/worst performers by Sharpe; the rest render as thin,
        # muted, unlabeled context lines (still fully visible, still hoverable
        # by name) — the complete numbers for every security remain in the
        # ranking table below regardless.
        MAX_HIGHLIGHTED = 8
        many_securities = len(rows) > MAX_HIGHLIGHTED
        highlighted: set[str] = set()
        if many_securities:
            by_sharpe = sorted(rows, key=lambda r: (r["Sharpe"] if r["Sharpe"] == r["Sharpe"] else -999), reverse=True)
            n_top = min(5, len(by_sharpe))
            n_bottom = min(3, len(by_sharpe) - n_top)
            highlighted = {r["Security"] for r in by_sharpe[:n_top]} | {r["Security"] for r in by_sharpe[-n_bottom:] if n_bottom > 0}

        equity_fig = go.Figure()
        for security, equity in equity_by_security.items():
            if many_securities and security not in highlighted:
                equity_fig.add_trace(go.Scatter(
                    x=equity.index, y=equity.values, mode="lines", name=security,
                    line=dict(width=1, color="#9ca3af"), opacity=0.35, showlegend=False,
                    hovertemplate=f"{security}: %{{y:.2f}}<extra></extra>",
                ))
        for security, equity in equity_by_security.items():
            if not many_securities or security in highlighted:
                equity_fig.add_trace(go.Scatter(
                    x=equity.index, y=equity.values, mode="lines", name=security,
                    line=dict(width=2.2), hovertemplate=f"{security}: %{{y:.2f}}<extra></extra>",
                ))
        title = "Growth of $1 — All Securities"
        subtitle = (
            f"{len(rows)} securities — legend highlights the {len(highlighted)} best/worst by Sharpe; "
            "hover any line to identify it; full ranking in the table below"
        ) if many_securities else None
        apply_theme(equity_fig, preset=params.get("theme", "professional"),
                    title=title, subtitle=subtitle, x_title="Date", y_title="Portfolio Value ($)",
                    height=520 if many_securities else 480)

        scatter_fig = go.Figure()
        label_text = [s if (not many_securities or s in highlighted) else "" for s in scatter_labels]
        scatter_fig.add_trace(go.Scatter(
            x=scatter_x, y=scatter_y, mode="markers+text" if any(label_text) else "markers",
            text=label_text, textposition="top center",
            hovertext=scatter_labels, hovertemplate="%{hovertext}<br>Vol %{x:.1f}% / CAGR %{y:.1f}%<extra></extra>",
            marker=dict(size=10 if many_securities else 12),
        ))
        apply_theme(scatter_fig, preset=params.get("theme", "professional"),
                    title="Risk / Return", subtitle=("Labeled points are the best/worst by Sharpe; hover any point to identify it" if many_securities else None),
                    x_title="Annualized Volatility (%)", y_title="CAGR (%)", height=420)

        ranking = sorted(rows, key=lambda r: (r["Sharpe"] if r["Sharpe"] == r["Sharpe"] else -999), reverse=True)
        best = ranking[0]
        stats = {
            "Securities Analyzed": len(rows),
            "Best Sharpe": f"{best['Security']} ({best['Sharpe']})",
            "Risk-Free Rate Used (%)": round(rf * 100, 2),
            "VaR/CVaR Confidence (%)": round(confidence * 100, 1),
        }

        return MethodResult(
            figure=self.fig_to_dict(equity_fig),
            stats=stats,
            tables={"ranking": ranking, "risk_return_scatter": self.fig_to_dict(scatter_fig)},
            series_csv_rows=ranking,
            warnings=(["Using unadjusted Close prices for at least one security — returns may be distorted around dividends/splits."]
                      if price_role_used == "close" else []),
        )
