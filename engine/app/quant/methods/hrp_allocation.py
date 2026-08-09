from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram

from app.data.reshape import ReshapeError, price_panel
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme
from app.quant.portfolio import hrp_linkage, hrp_weights, mean_returns_and_cov, portfolio_return, portfolio_vol


class HrpAllocationMethod(QuantMethod):
    id = "hrp_allocation"
    name = "Hierarchical Risk Parity (HRP)"
    category = "Portfolio Optimization"
    difficulty = "advanced"

    description = "Lopez de Prado's Hierarchical Risk Parity: allocates weights via correlation-based clustering and recursive risk-balancing, without inverting the covariance matrix."
    what_it_calculates = (
        "Portfolio weights that balance risk contribution across a hierarchy of asset clusters (built from "
        "correlation structure), rather than solving a mean-variance optimization that requires expected "
        "returns and a matrix inversion."
    )
    why_use_it = (
        "Mean-variance optimization (efficient frontier) is highly sensitive to expected-return estimates and "
        "can produce concentrated, unstable weights. HRP sidesteps both issues — it needs no expected-return "
        "input and avoids inverting a possibly ill-conditioned covariance matrix — at the cost of not directly "
        "targeting a specific expected return or Sharpe ratio."
    )
    methodology = (
        "1) Compute the correlation-distance $d_{ij}=\\sqrt{0.5(1-\\rho_{ij})}$ and run single-linkage "
        "hierarchical clustering. 2) Quasi-diagonalize the covariance matrix using the clustering's leaf "
        "order, placing similar assets adjacent. 3) Recursive bisection: repeatedly split the ordered asset "
        "list in half and allocate weight between the two halves inversely proportional to each half's "
        "(inverse-variance-weighted) cluster variance, so risk is equalized top-down through the hierarchy."
    )
    assumptions = [
        "Uses only the covariance/correlation matrix estimated from historical returns — no expected-return "
        "input is required or used, unlike mean-variance optimization.",
        "Single-linkage clustering on correlation-distance is one specific (the original Lopez de Prado) "
        "choice; other linkage methods would produce a different hierarchy and different weights.",
    ]
    limitations = [
        "HRP targets balanced risk contribution, not a specific expected return or maximum Sharpe ratio — its "
        "output is not directly comparable to the efficient frontier's optimal portfolios on those dimensions.",
        "Like any covariance-based method, results depend on the historical estimation window and are not "
        "guaranteed to hold going forward.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price", min_series=2,
                       note="At least 2 securities are required to build a hierarchy."),
    ]
    params = [
        ParamSpec("securities", "Securities (comma-separated, blank = all)", "string", default="",
                   description="Restrict the allocation to a subset of mapped securities."),
        ParamSpec("risk_free_rate", "Risk-free rate (annualized, %)", "float", default=0.0, min=-5, max=20,
                   description="Used only to report the resulting portfolio's Sharpe ratio."),
    ]

    def check_requirements(self, role_map, df):
        base = super().check_requirements(role_map, df)
        try:
            panel, _ = price_panel(df, role_map)
            if panel.shape[1] < 2:
                base.satisfied = False
                base.missing.append(f"Found only {panel.shape[1]} security — need at least 2 for HRP.")
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
            raise ValueError(f"Only {len(returns)} overlapping observations — need at least 30.")

        # HRP clusters assets by correlation-distance, which requires dividing
        # by each asset's standard deviation. A near-zero-variance security
        # (e.g. a money-market fund pegged at $1) makes that division blow up
        # into NaN/Inf, which scipy's linkage() rejects outright ("condensed
        # distance matrix must contain only finite values") — crashing the
        # whole calculation for every other, perfectly clusterable, security.
        # It also has no meaningful risk to balance against its peers, so
        # exclude it rather than let it break the run.
        std = returns.std(ddof=1)
        zero_variance = [s for s in securities if not np.isfinite(std[s]) or std[s] < 1e-12]
        dropped_warning = []
        if zero_variance:
            if len(securities) - len(zero_variance) < 2:
                raise ValueError(
                    f"{', '.join(zero_variance)} had ~zero return variance (e.g. a money-market fund pegged near "
                    "$1) and can't be clustered — too few remaining securities for HRP after excluding them."
                )
            dropped_warning.append(
                f"{', '.join(zero_variance)} had ~zero return variance over the overlapping window (e.g. a "
                "money-market fund pegged near $1.00) and can't be meaningfully correlation-clustered — excluded "
                "from the HRP allocation."
            )
            securities = [s for s in securities if s not in zero_variance]
            returns = returns[securities]

        weights = hrp_weights(returns)
        mean_ret, cov = mean_returns_and_cov(returns)
        rf = float(params.get("risk_free_rate", 0.0)) / 100.0

        port_ret = portfolio_return(weights.values, mean_ret.reindex(weights.index))
        port_vol = portfolio_vol(weights.values, cov.reindex(index=weights.index, columns=weights.index))
        sharpe = (port_ret - rf) / port_vol if port_vol else float("nan")

        corr = returns.corr()
        link = hrp_linkage(corr)
        dendro = dendrogram(link, labels=list(returns.columns), no_plot=True)

        dendro_fig = go.Figure()
        for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
            dendro_fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#2563eb", width=1.5), showlegend=False, hoverinfo="skip"))
        tick_positions = [5 + 10 * i for i in range(len(dendro["ivl"]))]
        dendro_fig.update_xaxes(tickmode="array", tickvals=tick_positions, ticktext=dendro["ivl"])
        apply_theme(dendro_fig, preset=params.get("theme", "professional"), title="Asset Clustering Dendrogram (correlation-distance)",
                    y_title="Distance", height=380)

        weights_sorted = weights.sort_values(ascending=False)
        weights_fig = go.Figure()
        weights_fig.add_trace(go.Bar(x=list(weights_sorted.index), y=weights_sorted.values * 100, marker_color="#2563eb"))
        apply_theme(weights_fig, preset=params.get("theme", "professional"), title="HRP Portfolio Weights",
                    x_title="Security", y_title="Weight (%)", height=380)

        stats = {
            "Securities": ", ".join(securities),
            "Observations Used": len(returns),
            "Portfolio Expected Return (annualized, %)": round(port_ret * 100, 2),
            "Portfolio Volatility (annualized, %)": round(port_vol * 100, 2),
            "Portfolio Sharpe Ratio": round(sharpe, 2) if sharpe == sharpe else None,
            "Largest Weight": f"{weights_sorted.index[0]} ({weights_sorted.iloc[0]*100:.1f}%)",
            "Smallest Weight": f"{weights_sorted.index[-1]} ({weights_sorted.iloc[-1]*100:.1f}%)",
        }

        rows = [{"security": sec, "weight_pct": round(float(weights[sec]) * 100, 3)} for sec in securities]

        warnings = dropped_warning[:]
        if price_role_used == "close":
            warnings.append("Using unadjusted Close prices for at least one security.")

        return MethodResult(
            figure=self.fig_to_dict(weights_fig),
            stats=stats,
            tables={"dendrogram": self.fig_to_dict(dendro_fig)},
            series_csv_rows=rows,
            warnings=warnings,
        )
