"""
Core quantitative math — pure functions, no plotting, no framework glue.

These are unit-tested directly against hand-calculated / known values
(see app/tests). Every QuantMethod implementation should call into this
module rather than re-deriving formulas inline, so there is exactly one
implementation of "what is a Sharpe ratio" in the whole codebase.

Annualization convention: 252 trading days/year for daily data unless the
caller passes a different `periods_per_year`. This is documented in every
method's `assumptions`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    p = prices.astype(float)
    return np.log(p / p.shift(1)).dropna()


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Growth of $1, from a simple-return series."""
    return (1.0 + returns).cumprod() - 1.0


def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    return initial_value * (1.0 + returns).cumprod()


def annualize_return(total_return: float, n_periods: int, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """CAGR from a total (holding-period) return over n_periods observations."""
    if n_periods <= 0:
        return float("nan")
    years = n_periods / periods_per_year
    if years <= 0:
        return float("nan")
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def annualize_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, mar: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    downside = returns[returns < mar] - mar
    if downside.empty:
        return 0.0
    return float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """risk_free_rate is annualized; converted to per-period before excess-return calc."""
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_per_period
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_per_period
    dd = downside_deviation(returns, mar=rf_per_period, periods_per_year=periods_per_year)
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float(excess.mean() * periods_per_year / dd)


def max_drawdown(cum_curve: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """cum_curve: an equity/index curve (not returns). Returns (max_dd, peak_date, trough_date) with max_dd <= 0."""
    running_max = cum_curve.cummax()
    dd = cum_curve / running_max - 1.0
    trough_idx = dd.idxmin()
    dd_val = float(dd.min()) if len(dd) else float("nan")
    peak_idx = cum_curve.loc[:trough_idx].idxmax() if trough_idx is not None and len(dd) else None
    return dd_val, peak_idx, trough_idx


def drawdown_series(cum_curve: pd.Series) -> pd.Series:
    running_max = cum_curve.cummax()
    return cum_curve / running_max - 1.0


def calmar_ratio(cagr: float, max_dd: float) -> float:
    if max_dd == 0 or np.isnan(max_dd):
        return float("nan")
    return float(cagr / abs(max_dd))


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) VaR as a positive loss magnitude at the given confidence level."""
    if returns.empty:
        return float("nan")
    q = np.percentile(returns, (1 - confidence) * 100)
    return float(-q)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    from scipy.stats import norm

    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    if returns.empty:
        return float("nan")
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= var_threshold]
    if tail.empty:
        return float(-var_threshold)
    return float(-tail.mean())


def ewma(series: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted moving average, span parameterization (like pandas .ewm(span=))."""
    return series.ewm(span=span, adjust=False).mean()


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics-style EWMA volatility (annualized), lam=decay factor."""
    var = returns.pow(2).ewm(alpha=(1 - lam), adjust=False).mean()
    return np.sqrt(var * TRADING_DAYS_PER_YEAR)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std(ddof=1)
    return (series - roll_mean) / roll_std


def skewness(returns: pd.Series) -> float:
    return float(returns.skew())


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (pandas default is already excess, i.e. normal = 0)."""
    return float(returns.kurt())
