"""
Portfolio-level mathematics: covariance/correlation, portfolio return/vol,
random-portfolio Monte Carlo search, and mean-variance optimization
(min-volatility / max-Sharpe / efficient frontier) via SciPy SLSQP.

Convention: `mean_returns` and `cov_matrix` are per-period (daily) unless
otherwise annotated; annualization is applied explicitly where needed so the
convention is visible at every call site rather than implicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.quant.calc import TRADING_DAYS_PER_YEAR


def mean_returns_and_cov(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """returns: wide DataFrame, one column per security, simple periodic returns."""
    clean = returns.dropna(how="any")
    return clean.mean(), clean.cov()


def portfolio_return(weights: np.ndarray, mean_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(np.dot(weights, mean_returns) * periods_per_year)


def portfolio_vol(weights: np.ndarray, cov_matrix: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    variance = float(weights @ cov_matrix.values @ weights)
    return float(np.sqrt(max(variance, 0.0)) * np.sqrt(periods_per_year))


def portfolio_sharpe(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                      risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    ret = portfolio_return(weights, mean_returns, periods_per_year)
    vol = portfolio_vol(weights, cov_matrix, periods_per_year)
    if vol == 0:
        return float("nan")
    return (ret - risk_free_rate) / vol


def random_portfolios(n_portfolios: int, n_assets: int, seed: int | None = None,
                       allow_short: bool = False) -> np.ndarray:
    """Vectorized generation of `n_portfolios` weight vectors summing to 1.
    Long-only: Dirichlet(1,...,1) gives a uniform distribution over the simplex."""
    rng = np.random.default_rng(seed)
    if allow_short:
        raw = rng.normal(size=(n_portfolios, n_assets))
        weights = raw / np.abs(raw).sum(axis=1, keepdims=True)
    else:
        weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)
    return weights


def simulate_random_portfolios(
    mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_portfolios: int = 5000,
    risk_free_rate: float = 0.0, seed: int | None = 7, allow_short: bool = False,
) -> pd.DataFrame:
    n_assets = len(mean_returns)
    weights = random_portfolios(n_portfolios, n_assets, seed=seed, allow_short=allow_short)
    cov_vals = cov_matrix.values
    mean_vals = mean_returns.values

    port_returns = weights @ mean_vals * TRADING_DAYS_PER_YEAR
    # vectorized quadratic form: (W @ Cov) elementwise* W, summed per row
    port_vars = np.einsum("ij,jk,ik->i", weights, cov_vals, weights)
    port_vols = np.sqrt(np.maximum(port_vars, 0.0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpes = np.where(port_vols > 0, (port_returns - risk_free_rate) / port_vols, np.nan)

    df = pd.DataFrame({"return": port_returns, "volatility": port_vols, "sharpe": sharpes})
    for i, asset in enumerate(mean_returns.index):
        df[f"weight_{asset}"] = weights[:, i]
    return df


def _weight_constraints(n_assets: int, allow_short: bool, max_weight: float | None, min_weight: float | None):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    lower = -1.0 if allow_short else (min_weight if min_weight is not None else 0.0)
    upper = max_weight if max_weight is not None else 1.0
    bounds = tuple((lower, upper) for _ in range(n_assets))
    return constraints, bounds


def optimize_min_volatility(mean_returns: pd.Series, cov_matrix: pd.DataFrame, allow_short: bool = False,
                             max_weight: float | None = None, min_weight: float | None = None) -> np.ndarray:
    n = len(mean_returns)
    constraints, bounds = _weight_constraints(n, allow_short, max_weight, min_weight)
    x0 = np.repeat(1.0 / n, n)
    result = minimize(lambda w: portfolio_vol(w, cov_matrix), x0, method="SLSQP",
                       bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Minimum-volatility optimization did not converge: {result.message}")
    return result.x


def optimize_max_sharpe(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.0,
                         allow_short: bool = False, max_weight: float | None = None,
                         min_weight: float | None = None) -> np.ndarray:
    n = len(mean_returns)
    constraints, bounds = _weight_constraints(n, allow_short, max_weight, min_weight)
    x0 = np.repeat(1.0 / n, n)

    def neg_sharpe(w):
        s = portfolio_sharpe(w, mean_returns, cov_matrix, risk_free_rate)
        return -s if s == s else 1e6  # guard NaN (zero-vol) from breaking the optimizer

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Maximum-Sharpe optimization did not converge: {result.message}")
    return result.x


def efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_points: int = 40,
                        allow_short: bool = False, max_weight: float | None = None,
                        min_weight: float | None = None) -> pd.DataFrame:
    """Trace the frontier by minimizing volatility at a grid of target returns
    between the min-vol portfolio's return and the max-return (single-asset) return."""
    n = len(mean_returns)
    min_vol_w = optimize_min_volatility(mean_returns, cov_matrix, allow_short, max_weight, min_weight)
    min_vol_ret = portfolio_return(min_vol_w, mean_returns)
    max_ret = float(mean_returns.max() * TRADING_DAYS_PER_YEAR)

    targets = np.linspace(min_vol_ret, max_ret, n_points)
    rows = []
    _, bounds = _weight_constraints(n, allow_short, max_weight, min_weight)
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mean_returns) - t},
        ]
        x0 = np.repeat(1.0 / n, n)
        result = minimize(lambda w: portfolio_vol(w, cov_matrix), x0, method="SLSQP",
                           bounds=bounds, constraints=constraints)
        if result.success:
            rows.append({"target_return": target, "volatility": portfolio_vol(result.x, cov_matrix),
                         "weights": result.x})
    return pd.DataFrame(rows)
