from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from app.data.reshape import price_panel
from app.quant import calc
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme
from app.quant.portfolio import resolve_security_series

MAX_MC_SIMS = 50000


class VarCvarMethod(QuantMethod):
    id = "var_cvar"
    name = "Value at Risk & Expected Shortfall"
    category = "Risk Analytics"
    difficulty = "intermediate"

    description = "Historical, parametric (variance-covariance), and Monte Carlo VaR alongside Expected Shortfall (CVaR), compared side by side."
    what_it_calculates = (
        "The loss threshold that historical/simulated returns exceed only (1-confidence) of the time (VaR), "
        "and the average loss in that worst tail (Expected Shortfall / CVaR), computed three ways for comparison."
    )
    why_use_it = (
        "VaR/ES are the standard regulatory and internal risk-budgeting tail-risk measures. Comparing methods "
        "surfaces how sensitive the estimate is to distributional assumptions."
    )
    methodology = (
        "Historical VaR: the empirical $(1-c)$ quantile of the historical return sample (no distributional "
        "assumption). Parametric VaR: assumes returns are Normal($\\mu,\\sigma$) and uses "
        "$VaR = -(\\mu + z_{1-c}\\sigma)$. Monte Carlo VaR: draws N simulated returns from a Normal($\\mu,\\sigma$) "
        "fit to the historical sample and takes the empirical quantile of the simulated sample (converges to "
        "the parametric result as N grows, since the same Normal assumption is used — this makes the "
        "Historical-vs-Normal-based gap the mathematically meaningful comparison here, not MC-vs-parametric). "
        "Expected Shortfall / CVaR is the mean loss conditional on being beyond the VaR threshold."
    )
    assumptions = [
        "Parametric and Monte Carlo VaR here assume Normally-distributed returns; historical VaR makes no "
        "distributional assumption but assumes the historical sample is representative of future risk.",
        "Single-period VaR at the return frequency of the uploaded data (not scaled to a different horizon).",
    ]
    limitations = [
        "Real return distributions typically have fatter tails than Normal, so parametric/Monte Carlo VaR under "
        "the Normal assumption tends to understate true tail risk.",
        "VaR does not describe the size of losses beyond the threshold — use Expected Shortfall for that.",
        "All three methods are backward-looking; none guarantees future losses will respect the estimated "
        "threshold, especially outside the historical sample's regime.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price"),
    ]
    params = [
        ParamSpec("security", "Security", "select", default=None, description="Security to analyze."),
        ParamSpec("confidence", "Confidence level (%)", "float", default=95, min=80, max=99.9,
                   description="E.g. 95% VaR is the loss exceeded only 5% of the time."),
        ParamSpec("n_sims", "Monte Carlo simulations", "int", default=10000, min=1000, max=MAX_MC_SIMS,
                   description="Number of draws for the Monte Carlo VaR estimate."),
        ParamSpec("seed", "Random seed", "int", default=42, min=0, max=2**31 - 1, description="For Monte Carlo reproducibility."),
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
        if len(prices) < 30:
            raise ValueError(f"Need at least 30 observations for a meaningful VaR estimate (found {len(prices)}).")

        returns = calc.simple_returns(prices)
        confidence = float(params.get("confidence", 95.0)) / 100.0
        seed = int(params.get("seed", 42))
        n_sims = min(int(params.get("n_sims", 10000)), MAX_MC_SIMS)

        hist_var = calc.historical_var(returns, confidence)
        hist_es = calc.expected_shortfall(returns, confidence)

        param_var = calc.parametric_var(returns, confidence)
        mu, sigma = float(returns.mean()), float(returns.std(ddof=1))
        z = norm.ppf(1 - confidence)
        param_es = float(-(mu - sigma * norm.pdf(z) / (1 - confidence)))

        rng = np.random.default_rng(seed)
        simulated = rng.normal(mu, sigma, n_sims)
        mc_var = float(-np.percentile(simulated, (1 - confidence) * 100))
        mc_tail = simulated[simulated <= np.percentile(simulated, (1 - confidence) * 100)]
        mc_es = float(-mc_tail.mean()) if len(mc_tail) else mc_var

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns.values * 100, nbinsx=60, name="Historical returns", opacity=0.75,
                                    histnorm="probability density"))
        for label, value, color in [
            (f"Historical VaR {int(confidence*100)}%", -hist_var * 100, "#dc2626"),
            (f"Parametric VaR {int(confidence*100)}%", -param_var * 100, "#f59e0b"),
            (f"Monte Carlo VaR {int(confidence*100)}%", -mc_var * 100, "#7c3aed"),
        ]:
            fig.add_vline(x=value, line=dict(color=color, dash="dash", width=2),
                          annotation_text=label, annotation_position="top")
        apply_theme(fig, preset=params.get("theme", "professional"),
                    title=f"{security} — Return Distribution & VaR ({int(confidence*100)}% confidence)",
                    x_title="Daily Return (%)", y_title="Density")

        stats = {
            "Observations": len(returns),
            "Confidence Level (%)": round(confidence * 100, 1),
            "Historical VaR (%)": round(hist_var * 100, 2),
            "Historical Expected Shortfall / CVaR (%)": round(hist_es * 100, 2),
            "Parametric VaR (Normal) (%)": round(param_var * 100, 2),
            "Parametric Expected Shortfall (%)": round(param_es * 100, 2),
            "Monte Carlo VaR (%)": round(mc_var * 100, 2),
            "Monte Carlo Expected Shortfall (%)": round(mc_es * 100, 2),
            "Mean Daily Return (%)": round(mu * 100, 4),
            "Daily Volatility (%)": round(sigma * 100, 4),
        }

        comparison = [
            {"Method": "Historical", "VaR (%)": round(hist_var * 100, 2), "Expected Shortfall (%)": round(hist_es * 100, 2)},
            {"Method": "Parametric (Normal)", "VaR (%)": round(param_var * 100, 2), "Expected Shortfall (%)": round(param_es * 100, 2)},
            {"Method": "Monte Carlo", "VaR (%)": round(mc_var * 100, 2), "Expected Shortfall (%)": round(mc_es * 100, 2)},
        ]

        warnings = []
        if price_role_used == "close":
            warnings.append("Using unadjusted Close prices.")
        skew = calc.skewness(returns)
        if abs(skew) > 1 or calc.kurtosis(returns) > 3:
            warnings.append(
                "Returns show substantial skew/excess kurtosis — parametric and Monte Carlo VaR (which assume "
                "Normal returns) likely understate true tail risk here. Prefer the historical estimate."
            )

        return MethodResult(
            figure=self.fig_to_dict(fig),
            stats=stats,
            tables={"method_comparison": comparison},
            series_csv_rows=comparison,
            warnings=warnings,
        )
