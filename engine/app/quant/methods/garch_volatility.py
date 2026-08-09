from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from arch import arch_model

from app.data.reshape import price_panel
from app.quant import calc
from app.quant.base import MethodResult, ParamSpec, QuantMethod, RequiredInput
from app.quant.chart_theme import apply_theme
from app.quant.portfolio import resolve_security_series


class GarchVolatilityMethod(QuantMethod):
    id = "garch_volatility"
    name = "GARCH Volatility Forecasting"
    category = "Statistical / Econometric"
    difficulty = "advanced"

    description = "Fits a GARCH(p,q) model to capture volatility clustering, and forecasts conditional volatility forward."
    what_it_calculates = (
        "A GARCH(p,q) conditional-volatility model fit to daily returns, showing the in-sample conditional "
        "volatility (which rises and falls with volatility clustering, unlike a flat historical-average "
        "estimate) plus a forward volatility forecast."
    )
    why_use_it = (
        "Volatility clusters in real markets — calm periods and turbulent periods persist. GARCH captures this "
        "directly, unlike a single trailing-window volatility estimate, and is standard for risk forecasting "
        "and options-adjacent volatility work."
    )
    methodology = (
        "GARCH(p,q): $\\sigma_t^2 = \\omega + \\sum_{i=1}^{q}\\alpha_i \\epsilon_{t-i}^2 + \\sum_{j=1}^{p}\\beta_j \\sigma_{t-j}^2$, "
        "fit by maximum likelihood (via the `arch` package) on daily returns scaled by 100 for numerical "
        "stability. Forecasts are the model's analytic multi-step-ahead conditional-variance forecast, "
        "annualized by $\\times\\sqrt{252}$."
    )
    assumptions = [
        "Assumes a GARCH(p,q) structure is an adequate description of the volatility process; higher-order or "
        "asymmetric (EGARCH/GJR) effects are not modeled here.",
        "Return innovations are assumed to follow the selected distribution (Normal or Student-t).",
        "The forecast assumes no structural break in the volatility process beyond the sample used to fit.",
    ]
    limitations = [
        "GARCH forecasts revert toward the model's long-run variance and become less informative at longer "
        "horizons.",
        "Model fit quality depends heavily on sample length; short samples produce unstable parameter "
        "estimates.",
        "This forecasts volatility, not direction — it says nothing about expected returns.",
    ]

    required_inputs = [
        RequiredInput(role="date", label="Date"),
        RequiredInput(role="close", label="Close or Adjusted Close price"),
    ]
    params = [
        ParamSpec("security", "Security", "select", default=None, description="Security to model."),
        ParamSpec("p", "GARCH lag (p)", "int", default=1, min=1, max=3, description="Number of lagged conditional-variance terms."),
        ParamSpec("q", "ARCH lag (q)", "int", default=1, min=1, max=3, description="Number of lagged squared-residual terms."),
        ParamSpec("distribution", "Innovation distribution", "select", default="normal",
                   options=[{"value": "normal", "label": "Normal"}, {"value": "t", "label": "Student-t (fat tails)"}],
                   description="Distributional assumption for return innovations."),
        ParamSpec("forecast_days", "Forecast horizon (days)", "int", default=20, min=1, max=120,
                   description="Number of trading days to forecast conditional volatility forward."),
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
        returns = calc.simple_returns(prices)
        if len(returns) < 100:
            raise ValueError(f"GARCH estimation needs at least 100 return observations (found {len(returns)}).")

        p = int(params.get("p", 1))
        q = int(params.get("q", 1))
        dist = params.get("distribution", "normal")
        horizon = int(params.get("forecast_days", 20))

        scaled_returns = returns.values * 100  # arch recommends O(1-100) scale for numerical stability
        model = arch_model(scaled_returns, vol="GARCH", p=p, q=q, dist=dist, rescale=False)
        fit = model.fit(disp="off")

        cond_vol_daily = fit.conditional_volatility / 100.0
        cond_vol_annualized = cond_vol_daily * np.sqrt(calc.TRADING_DAYS_PER_YEAR)
        cond_vol_series = pd.Series(cond_vol_annualized, index=returns.index)

        realized_vol = returns.rolling(21).std(ddof=1) * np.sqrt(calc.TRADING_DAYS_PER_YEAR)

        forecast = fit.forecast(horizon=horizon, reindex=False)
        forecast_var_daily = forecast.variance.values[-1] / 100.0**2
        forecast_vol_annualized = np.sqrt(forecast_var_daily) * np.sqrt(calc.TRADING_DAYS_PER_YEAR)
        last_date = returns.index[-1]
        forecast_dates = pd.bdate_range(last_date, periods=horizon + 1)[1:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cond_vol_series.index, y=cond_vol_series.values * 100, name="GARCH Conditional Volatility (annualized %)", line=dict(width=1.6)))
        fig.add_trace(go.Scatter(x=realized_vol.index, y=realized_vol.values * 100, name="21-day Realized Volatility (%)",
                                  line=dict(width=1, dash="dot", color="#94a3b8")))
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_vol_annualized * 100, name=f"{horizon}-day Forecast",
                                  line=dict(width=2, color="#dc2626", dash="dash")))
        apply_theme(fig, preset=params.get("theme", "professional"),
                    title=f"{security} — GARCH({p},{q}) Conditional Volatility & {horizon}-day Forecast",
                    subtitle=f"{dist} innovations", x_title="Date", y_title="Annualized Volatility (%)")

        stats = {
            "Model": f"GARCH({p},{q}), {dist} innovations",
            "Log-Likelihood": round(float(fit.loglikelihood), 2),
            "AIC": round(float(fit.aic), 2),
            "BIC": round(float(fit.bic), 2),
            "Current Conditional Volatility (annualized, %)": round(float(cond_vol_series.iloc[-1]) * 100, 2),
            f"{horizon}-day Forecast Volatility (annualized, %)": round(float(forecast_vol_annualized[-1]) * 100, 2),
            "Historical (unconditional) Volatility (%)": round(calc.annualize_vol(returns) * 100, 2),
            "Observations": len(returns),
        }

        rows = [
            {"date": str(d.date()), "conditional_volatility_annualized": float(cond_vol_series.get(d, np.nan))}
            for d in cond_vol_series.index
        ] + [
            {"date": str(d.date()), "forecast_volatility_annualized": float(v)}
            for d, v in zip(forecast_dates, forecast_vol_annualized)
        ]

        warnings = []
        if price_role_used == "close":
            warnings.append("Using unadjusted Close prices.")
        if not fit.convergence_flag == 0:
            warnings.append("The GARCH optimizer did not fully converge — parameter estimates may be unreliable.")

        return MethodResult(
            figure=self.fig_to_dict(fig),
            stats=stats,
            tables={},
            series_csv_rows=rows,
            warnings=warnings,
        )
