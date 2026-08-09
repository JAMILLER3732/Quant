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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from app.quant.calc import TRADING_DAYS_PER_YEAR


def mean_returns_and_cov(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """returns: wide DataFrame, one column per security, simple periodic returns."""
    clean = returns.dropna(how="any")
    return clean.mean(), clean.cov()


# Sentinel selectable from every single-security "security" dropdown (via
# check_requirements' dynamic_param_options) that means "analyze the whole
# portfolio, not one holding." Deliberately a display-ready string (not an
# opaque id) since the frontend renders dropdown options verbatim.
PORTFOLIO_LABEL = "◆ Whole Portfolio (all holdings, equal-weight, daily-rebalanced)"


def blended_portfolio_series(panel: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """
    Collapse a wide multi-security price panel into one blended portfolio
    price series, for feeding into single-series methods (returns/VaR/GARCH/
    regime/etc.) so they can analyze "the portfolio" rather than one holding.

    Methodology: daily simple returns per security -> weighted average return
    each day (equal-weight by default) -> compounded into an index starting
    at 100. This is a *daily-rebalanced* blend (the standard, simplest
    definition of a blended benchmark/index) — it is not a buy-and-hold
    portfolio of fixed share counts, which would additionally require
    knowing entry prices/quantities that a plain price panel doesn't carry.
    Only dates where every included security has a price are used, so the
    blend is never silently built from a partial/shifting basket.
    """
    clean = panel.dropna(how="any")
    if clean.shape[0] < 2:
        raise ValueError(
            "Not enough overlapping dates across all securities to build a blended portfolio series "
            f"(found {clean.shape[0]})."
        )
    securities = list(clean.columns)
    if weights is None:
        w = pd.Series(1.0 / len(securities), index=securities)
    else:
        w = pd.Series({s: weights.get(s, 0.0) for s in securities})
        total = w.sum()
        w = w / total if total > 0 else pd.Series(1.0 / len(securities), index=securities)

    security_returns = clean.pct_change().dropna(how="all")
    portfolio_returns = (security_returns * w).sum(axis=1)
    index_series = 100.0 * (1.0 + portfolio_returns).cumprod()
    index_series.name = "Portfolio"
    return index_series


def resolve_security_series(
    panel: pd.DataFrame, selection: str | None, weights: dict[str, float] | None = None
) -> tuple[pd.Series, str]:
    """
    Resolve a "security" param value to a price series + display name: either
    one column of the panel, or (if `selection` is the PORTFOLIO_LABEL
    sentinel, missing, or not an actual column and more than one security is
    available) the blended whole-portfolio series.
    """
    if selection in panel.columns:
        return panel[selection].dropna(), str(selection)
    if panel.shape[1] >= 2:
        return blended_portfolio_series(panel, weights), "Portfolio"
    # single-security panel with no valid selection — just use the one column
    only = panel.columns[0]
    return panel[only].dropna(), str(only)


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


def hrp_linkage(corr: pd.DataFrame) -> np.ndarray:
    """Lopez de Prado's correlation-distance: d_ij = sqrt(0.5*(1-corr_ij)), then
    single-linkage hierarchical clustering on the condensed distance matrix."""
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method="single")


def hrp_quasi_diag_order(link: np.ndarray, n_assets: int) -> list[int]:
    """Recover the leaf order (quasi-diagonalization) from a linkage matrix via
    scipy's own dendrogram leaf-ordering, avoiding a hand-rolled recursive walk."""
    return dendrogram(link, no_plot=True)["leaves"]


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity (Lopez de Prado, 2016):
      1. Cluster assets by correlation-distance (single linkage).
      2. Quasi-diagonalize the covariance matrix by the cluster leaf order.
      3. Recursive bisection: split the ordered list in half repeatably, and at
         each split allocate inversely proportional to each half's cluster
         variance (inverse-variance weighting within a cluster), so risk is
         balanced across the hierarchy rather than across raw asset counts.
    """
    cov = returns.cov()
    corr = returns.corr()
    assets = list(returns.columns)
    n = len(assets)

    link = hrp_linkage(corr)
    order = hrp_quasi_diag_order(link, n)
    ordered_assets = [assets[i] for i in order]

    weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]

    def cluster_variance(items: list[str]) -> float:
        sub_cov = cov.loc[items, items]
        inv_diag = 1.0 / np.diag(sub_cov.values)
        ivp = inv_diag / inv_diag.sum()  # inverse-variance weights within the cluster
        return float(ivp @ sub_cov.values @ ivp)

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            var_left, var_right = cluster_variance(left), cluster_variance(right)
            alloc_left = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alloc_left
            weights[right] *= (1.0 - alloc_left)
            new_clusters.extend([left, right])
        clusters = new_clusters

    return weights.reindex(assets)


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
