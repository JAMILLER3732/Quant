from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.data.reshape import ReshapeError, price_panel
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme


class CorrelationAnalysisMethod(QuantMethod):
    id = "correlation_analysis"
    name = "Correlation & Covariance Analysis"
    category = "Correlation & Dependence"
    difficulty = "beginner"

    description = "Correlation/covariance matrix heatmap plus a rolling pairwise-correlation chart across your mapped securities."
    what_it_calculates = (
        "The full pairwise Pearson (or Spearman rank) correlation and covariance matrices of daily returns "
        "across all selected securities, and a rolling-window correlation time series for one chosen pair."
    )
    why_use_it = (
        "Diversification depends on correlation, not just individual volatility. This is the standard first "
        "check before combining securities into a portfolio or building a pairs-trade."
    )
    methodology = (
        "Pearson correlation: $\\rho_{ij} = \\text{Cov}(r_i, r_j) / (\\sigma_i \\sigma_j)$ on simple daily "
        "returns. Spearman uses rank-transformed returns, capturing monotonic (not necessarily linear) "
        "dependence and is more robust to outliers. Rolling correlation recomputes $\\rho_{ij}$ over a trailing "
        "window to show how the relationship evolves through time."
    )
    assumptions = [
        "Only dates where all selected securities have simple returns available (the overlapping window) are "
        "used — securities with different histories are implicitly truncated to their common range.",
        "Pearson correlation measures linear dependence only; use Spearman if the relationship may be "
        "monotonic-but-nonlinear or the data contains outliers.",
    ]
    limitations = [
        "Correlation is not stationary — historical correlation, especially over long windows, may not predict "
        "future co-movement (correlations are well known to rise in market stress, reducing realized "
        "diversification exactly when it's needed most).",
        "Correlation does not imply causation, and a low sample size makes correlation estimates noisy.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price", min_series=2,
                       note="At least 2 securities are required for a correlation matrix."),
    ]
    params = [
        ParamSpec("securities", "Securities (comma-separated, blank = all)", "string", default="",
                   description="Restrict the matrix to a subset of mapped securities."),
        ParamSpec("method", "Correlation method", "select", default="pearson",
                   options=[{"value": "pearson", "label": "Pearson (linear)"},
                            {"value": "spearman", "label": "Spearman (rank)"},
                            {"value": "kendall", "label": "Kendall (rank, robust)"}],
                   description="Pearson measures linear correlation; Spearman/Kendall measure rank/monotonic dependence."),
        ParamSpec("rolling_window", "Rolling correlation window (days)", "int", default=60, min=10, max=252,
                   description="Trailing window for the rolling-correlation time series."),
        ParamSpec("pair_a", "Rolling pair — security A", "string", default="", description="First security for the rolling-correlation chart."),
        ParamSpec("pair_b", "Rolling pair — security B", "string", default="", description="Second security for the rolling-correlation chart."),
    ]

    def check_requirements(self, role_map, df):
        base = super().check_requirements(role_map, df)
        try:
            panel, _ = price_panel(df, role_map)
            if panel.shape[1] < 2:
                base.satisfied = False
                base.missing.append(f"Found only {panel.shape[1]} security — need at least 2 for correlation.")
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
        if len(returns) < 10:
            raise ValueError(f"Only {len(returns)} overlapping observations — need at least 10.")

        method = params.get("method", "pearson")
        corr = returns.corr(method=method)
        cov = returns.cov()

        # Per-cell numeric labels and 60px-per-row heights work for a handful of
        # securities but become unreadable clutter (and a runaway page height —
        # 60px * 82 securities is ~5000px) on a real multi-dozen-security
        # portfolio. Past a size threshold, drop the cell text (exact values
        # are still one hover away) and cap the height so the matrix stays
        # scannable instead of ballooning; tick labels shrink and rotate so
        # they stop colliding with each other along the x-axis.
        n = len(securities)
        dense = n > 25
        heatmap = go.Figure(data=go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate=None if dense else "%{text}",
            hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
            colorbar=dict(title=f"{method.title()} ρ"),
        ))
        apply_theme(heatmap, preset=params.get("theme", "professional"),
                    title=f"{method.title()} Correlation Matrix",
                    subtitle=f"{n} securities — hover any cell for its exact value" if dense else None,
                    height=min(max(360, 26 * n), 1400) if dense else max(360, 60 * n))
        heatmap.update_xaxes(tickangle=-45, tickfont=dict(size=max(7, min(12, 480 // max(n, 1)))))
        heatmap.update_yaxes(tickfont=dict(size=max(7, min(12, 480 // max(n, 1)))))

        pair_a = params.get("pair_a") or securities[0]
        pair_b = params.get("pair_b") or (securities[1] if len(securities) > 1 else securities[0])
        window = int(params.get("rolling_window", 60))
        roll_fig = go.Figure()
        if pair_a in returns.columns and pair_b in returns.columns and pair_a != pair_b:
            rolling_corr = returns[pair_a].rolling(window).corr(returns[pair_b])
            roll_fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values, mode="lines",
                                           name=f"Corr({pair_a}, {pair_b})", line=dict(width=1.8)))
            roll_fig.add_hline(y=0, line=dict(color="#94a3b8", width=1))
        apply_theme(roll_fig, preset=params.get("theme", "professional"),
                    title=f"Rolling {window}-day Correlation: {pair_a} vs {pair_b}",
                    x_title="Date", y_title="Correlation", height=340)

        offdiag = corr.values[~np.eye(len(securities), dtype=bool)]
        # A single near-zero-variance security (e.g. a money-market fund pegged
        # at $1.00) makes its correlation with everything else NaN (0/0). Using
        # plain mean/max/min would let that one security's NaNs silently null
        # out the summary stats for the *entire* matrix, even when every other
        # pair is perfectly valid — use nan-aware reducers instead, and name
        # the offending securities so the user knows why they're excluded.
        zero_variance = [s for s in securities if returns[s].std(ddof=1) < 1e-12 or not np.isfinite(returns[s].std(ddof=1))]
        valid_offdiag = offdiag[~np.isnan(offdiag)]
        stats = {
            "Securities": ", ".join(securities),
            "Observations Used": len(returns),
            "Method": method.title(),
            "Average Pairwise Correlation": round(float(valid_offdiag.mean()), 3) if len(valid_offdiag) else None,
            "Max Pairwise Correlation": round(float(valid_offdiag.max()), 3) if len(valid_offdiag) else None,
            "Min Pairwise Correlation": round(float(valid_offdiag.min()), 3) if len(valid_offdiag) else None,
        }

        corr_rows = [{"security": idx, **{c: round(float(v), 4) if np.isfinite(v) else None for c, v in row.items()}} for idx, row in corr.iterrows()]
        cov_table = [{"security": idx, **{c: round(float(v), 6) if np.isfinite(v) else None for c, v in row.items()}} for idx, row in cov.iterrows()]

        warnings = []
        if price_role_used == "close":
            warnings.append("Using unadjusted Close prices for at least one security.")
        if zero_variance:
            warnings.append(
                f"{', '.join(zero_variance)} had ~zero return variance over the overlapping window "
                "(e.g. a money-market fund pegged near $1.00) — its correlation with every other security is "
                "undefined (0/0) and shown as blank in the matrix; excluded from the summary stats above."
            )

        return MethodResult(
            figure=self.fig_to_dict(heatmap),
            stats=stats,
            tables={"rolling_correlation": self.fig_to_dict(roll_fig), "covariance_matrix": cov_table},
            series_csv_rows=corr_rows,
            warnings=warnings,
        )
