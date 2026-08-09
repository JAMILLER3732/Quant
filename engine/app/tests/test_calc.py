"""
Quantitative-calculation tests: every core formula in app/quant/calc.py is
checked against hand-calculated or independently-derived expected values,
per the project's mandatory quantitative-testing requirement.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.quant import calc


def _series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_simple_returns_hand_calculated():
    prices = _series([100, 110, 121])
    r = calc.simple_returns(prices)
    assert r.iloc[0] == pytest.approx(0.10, abs=1e-9)
    assert r.iloc[1] == pytest.approx(0.10, abs=1e-9)


def test_log_returns_hand_calculated():
    prices = _series([100, 110, 121])
    r = calc.log_returns(prices)
    assert r.iloc[0] == pytest.approx(math.log(1.1), abs=1e-9)
    assert r.iloc[1] == pytest.approx(math.log(1.1), abs=1e-9)


def test_cumulative_returns_compound_correctly():
    returns = _series([0.10, 0.10])
    cum = calc.cumulative_returns(returns)
    # (1.1 * 1.1) - 1 = 0.21
    assert cum.iloc[-1] == pytest.approx(0.21, abs=1e-9)


def test_equity_curve_starts_at_initial_value():
    returns = _series([0.10, -0.05])
    eq = calc.equity_curve(returns, initial_value=100.0)
    assert eq.iloc[0] == pytest.approx(110.0)
    assert eq.iloc[1] == pytest.approx(110.0 * 0.95)


def test_annualize_return_known_example():
    # Two periods of +10% each, with periods_per_year=1 means "2 years" of compounding
    # total_return = 1.1*1.1 - 1 = 0.21 -> CAGR over 2 years = sqrt(1.21) - 1 = 0.10
    cagr = calc.annualize_return(total_return=0.21, n_periods=2, periods_per_year=1)
    assert cagr == pytest.approx(0.10, abs=1e-9)


def test_annualize_vol_matches_manual_stdev():
    returns = _series([0.01, -0.02, 0.015, 0.005, -0.01])
    result = calc.annualize_vol(returns, periods_per_year=252)
    expected = returns.std(ddof=1) * math.sqrt(252)
    assert result == pytest.approx(expected, abs=1e-9)


def test_sharpe_ratio_zero_rf_matches_manual_formula():
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.00, -0.02, 0.015])
    sr = calc.sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    expected = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
    assert sr == pytest.approx(expected, abs=1e-9)


def test_sharpe_ratio_of_zero_volatility_is_nan():
    returns = _series([0.01, 0.01, 0.01, 0.01])
    sr = calc.sharpe_ratio(returns)
    assert math.isnan(sr)


def test_sortino_ignores_upside_deviation():
    # All positive returns -> no downside deviation -> Sortino should be inf-ish (nan-guarded to nan here since dd=0)
    returns = _series([0.01, 0.02, 0.03])
    sortino = calc.sortino_ratio(returns, risk_free_rate=0.0)
    assert math.isnan(sortino)  # downside deviation is 0 -> guarded, matches implementation contract

    # Mixed returns: downside deviation should only reflect the negative one
    returns2 = _series([0.02, -0.01, 0.015, -0.005])
    dd = calc.downside_deviation(returns2, mar=0.0, periods_per_year=252)
    downside_only = returns2[returns2 < 0]
    expected_dd = math.sqrt((downside_only**2).mean()) * math.sqrt(252)
    assert dd == pytest.approx(expected_dd, abs=1e-9)


def test_max_drawdown_hand_calculated():
    # equity curve: peak at 1.10 (t=1), trough at 0.90 (t=3) -> dd = 0.90/1.10 - 1
    curve = _series([1.00, 1.10, 1.05, 0.90, 1.20])
    dd, peak, trough = calc.max_drawdown(curve)
    assert dd == pytest.approx(0.90 / 1.10 - 1, abs=1e-9)
    assert peak == curve.index[1]
    assert trough == curve.index[3]


def test_drawdown_series_is_never_positive():
    curve = _series([1.0, 1.2, 1.1, 1.3, 0.8])
    dd = calc.drawdown_series(curve)
    assert (dd <= 1e-12).all()
    assert dd.iloc[0] == pytest.approx(0.0)


def test_calmar_ratio_hand_calculated():
    assert calc.calmar_ratio(cagr=0.12, max_dd=-0.20) == pytest.approx(0.6, abs=1e-9)


def test_historical_var_matches_numpy_percentile():
    returns = _series(list(np.linspace(-0.05, 0.05, 101)))
    var95 = calc.historical_var(returns, confidence=0.95)
    expected = -np.percentile(returns.values, 5)
    assert var95 == pytest.approx(expected, abs=1e-9)


def test_expected_shortfall_is_worse_than_var():
    returns = _series(list(np.linspace(-0.10, 0.05, 200)))
    var95 = calc.historical_var(returns, confidence=0.95)
    es95 = calc.expected_shortfall(returns, confidence=0.95)
    assert es95 >= var95  # tail mean loss should be at least as large as the quantile loss


def test_ewma_matches_manual_recursion():
    values = _series([10, 11, 9, 12, 13])
    span = 3
    alpha = 2 / (span + 1)
    result = calc.ewma(values, span)

    manual = [values.iloc[0]]
    for v in values.iloc[1:]:
        manual.append(alpha * v + (1 - alpha) * manual[-1])

    np.testing.assert_allclose(result.values, manual, atol=1e-9)


def test_rolling_zscore_hand_calculated():
    values = _series([1, 2, 3, 4, 100])  # last point should be a big outlier
    z = calc.rolling_zscore(values, window=4)
    window_vals = values.iloc[1:5]  # window covering index 1..4 for the last point
    expected_last = (values.iloc[4] - window_vals.mean()) / window_vals.std(ddof=1)
    assert z.iloc[4] == pytest.approx(expected_last, abs=1e-9)
    assert z.iloc[4] > 1  # confirms it's flagged as an above-average move


def test_skew_and_kurtosis_of_symmetric_distribution_near_zero():
    rng = np.random.default_rng(42)
    normal_returns = _series(list(rng.normal(0, 0.01, 5000)))
    assert abs(calc.skewness(normal_returns)) < 0.15
    assert abs(calc.kurtosis(normal_returns)) < 0.3
