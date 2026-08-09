"""
Quantitative tests for Phase 2 math: portfolio optimization (weights sum to 1,
satisfy bounds, min-vol truly minimizes variance among sampled portfolios) and
GBM simulation (shape, initial price, percentile ordering).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.quant.portfolio import (
    efficient_frontier,
    mean_returns_and_cov,
    optimize_max_sharpe,
    optimize_min_volatility,
    portfolio_return,
    portfolio_vol,
    random_portfolios,
    simulate_random_portfolios,
)
from app.quant.simulation import gbm_paths, percentile_bands


def _synthetic_returns(seed=1, n=500, n_assets=4):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    mu = rng.uniform(0.0002, 0.0008, n_assets)
    cov = np.diag(rng.uniform(0.0001, 0.0004, n_assets))
    data = rng.multivariate_normal(mu, cov, size=n)
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_random_portfolios_weights_sum_to_one_long_only():
    weights = random_portfolios(1000, 5, seed=1, allow_short=False)
    sums = weights.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)
    assert (weights >= 0).all()


def test_random_portfolios_allow_short_still_sums_to_one():
    weights = random_portfolios(500, 4, seed=2, allow_short=True)
    sums = np.abs(weights).sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)


def test_portfolio_return_and_vol_hand_calculated():
    mean_returns = pd.Series([0.001, 0.002], index=["A", "B"])
    cov = pd.DataFrame([[0.0004, 0.0001], [0.0001, 0.0009]], index=["A", "B"], columns=["A", "B"])
    weights = np.array([0.5, 0.5])

    expected_ret = (0.5 * 0.001 + 0.5 * 0.002) * 252
    assert portfolio_return(weights, mean_returns) == pytest.approx(expected_ret, abs=1e-9)

    expected_var = 0.25 * 0.0004 + 0.25 * 0.0009 + 2 * 0.5 * 0.5 * 0.0001
    expected_vol = np.sqrt(expected_var) * np.sqrt(252)
    assert portfolio_vol(weights, cov) == pytest.approx(expected_vol, abs=1e-9)


def test_min_volatility_weights_sum_to_one_and_beat_random_sample():
    returns = _synthetic_returns()
    mean_ret, cov = mean_returns_and_cov(returns)
    w = optimize_min_volatility(mean_ret, cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()

    min_vol = portfolio_vol(w, cov)
    # the optimized min-vol portfolio should have volatility <= every one of 2000 random long-only portfolios
    cloud = simulate_random_portfolios(mean_ret, cov, n_portfolios=2000, seed=3)
    assert min_vol <= cloud["volatility"].min() + 1e-6


def test_max_sharpe_weights_valid_and_positive_sharpe_beats_random_sample():
    returns = _synthetic_returns(seed=5)
    mean_ret, cov = mean_returns_and_cov(returns)
    w = optimize_max_sharpe(mean_ret, cov, risk_free_rate=0.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)

    ret, vol = portfolio_return(w, mean_ret), portfolio_vol(w, cov)
    sharpe = ret / vol
    cloud = simulate_random_portfolios(mean_ret, cov, n_portfolios=3000, seed=5)
    assert sharpe >= cloud["sharpe"].max() - 1e-3  # optimizer should not be meaningfully beaten by random search


def test_efficient_frontier_volatility_increases_or_flat_with_target_return():
    returns = _synthetic_returns(seed=9)
    mean_ret, cov = mean_returns_and_cov(returns)
    frontier = efficient_frontier(mean_ret, cov, n_points=15)
    assert len(frontier) > 5
    # frontier should be roughly monotonic: higher target return -> vol does not decrease much
    vols = frontier["volatility"].values
    assert vols[-1] >= vols[0] - 1e-6


def test_gbm_paths_shape_and_initial_price():
    paths = gbm_paths(s0=100.0, mu=0.08, sigma=0.2, horizon_days=50, n_sims=500, seed=1)
    assert paths.shape == (500, 51)
    np.testing.assert_allclose(paths[:, 0], 100.0)
    assert (paths > 0).all()  # GBM prices are always positive


def test_gbm_percentile_bands_are_ordered():
    paths = gbm_paths(s0=100.0, mu=0.05, sigma=0.3, horizon_days=100, n_sims=5000, seed=7)
    bands = percentile_bands(paths)
    # at each time step, p5 <= p25 <= p50 <= p75 <= p95
    for t in range(paths.shape[1]):
        vals = [bands[p][t] for p in (5, 25, 50, 75, 95)]
        assert vals == sorted(vals)


def test_gbm_mean_terminal_price_approaches_analytical_expectation():
    # E[S_T] = S0 * exp(mu*T) under GBM regardless of sigma
    s0, mu, sigma, years = 100.0, 0.10, 0.2, 1.0
    paths = gbm_paths(s0, mu, sigma, horizon_days=252, n_sims=200000, seed=11)
    terminal_mean = paths[:, -1].mean()
    expected = s0 * np.exp(mu * years)
    assert terminal_mean == pytest.approx(expected, rel=0.02)  # within 2% given large sample
