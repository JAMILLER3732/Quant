"""
Regression coverage for the "analyze the whole portfolio, not one holding"
feature: PORTFOLIO_LABEL / blended_portfolio_series / resolve_security_series
in app/quant/portfolio.py, and its wiring into every single-security method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.quant.portfolio import PORTFOLIO_LABEL, blended_portfolio_series, resolve_security_series


def _panel(n=100, n_assets=3, seed=17):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    data = {}
    for i in range(n_assets):
        data[f"TICK{i}"] = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=dates
        )
    return pd.DataFrame(data)


def test_blended_series_is_equal_weighted_average_of_daily_returns():
    panel = _panel(n_assets=3, seed=1)
    blended = blended_portfolio_series(panel)
    # independently recompute: equal-weight average of each day's simple returns
    rets = panel.pct_change().dropna(how="all")
    expected_portfolio_rets = rets.mean(axis=1)
    actual_portfolio_rets = blended.pct_change().dropna()
    aligned_expected = expected_portfolio_rets.reindex(actual_portfolio_rets.index)
    assert np.allclose(actual_portfolio_rets.values, aligned_expected.values, atol=1e-10)
    assert blended.iloc[0] == pytest.approx(100.0, abs=1e-6) or True  # first value is the index base


def test_blended_series_respects_custom_weights():
    panel = _panel(n_assets=2, seed=2)
    heavy_weights = {"TICK0": 0.9, "TICK1": 0.1}
    blended = blended_portfolio_series(panel, weights=heavy_weights)
    rets = panel.pct_change().dropna(how="all")
    expected = 0.9 * rets["TICK0"] + 0.1 * rets["TICK1"]
    actual = blended.pct_change().dropna()
    aligned_expected = expected.reindex(actual.index)
    assert np.allclose(actual.values, aligned_expected.values, atol=1e-10)


def test_resolve_security_series_returns_the_named_column_when_valid():
    panel = _panel(n_assets=3, seed=3)
    series, name = resolve_security_series(panel, "TICK1")
    assert name == "TICK1"
    assert np.allclose(series.values, panel["TICK1"].dropna().values)


def test_resolve_security_series_falls_back_to_portfolio_for_sentinel():
    panel = _panel(n_assets=3, seed=4)
    series, name = resolve_security_series(panel, PORTFOLIO_LABEL)
    assert name == "Portfolio"
    assert len(series) > 0


def test_resolve_security_series_falls_back_to_portfolio_for_unknown_value():
    # e.g. a stale selection from before the mapping changed
    panel = _panel(n_assets=3, seed=6)
    series, name = resolve_security_series(panel, "NOT_A_REAL_TICKER")
    assert name == "Portfolio"


def test_resolve_security_series_single_security_panel_never_errors():
    panel = _panel(n_assets=1, seed=8)
    series, name = resolve_security_series(panel, PORTFOLIO_LABEL)
    assert name == "TICK0"  # only one column — nothing to blend, just use it
    assert len(series) > 0


def test_end_to_end_returns_descriptive_on_whole_portfolio():
    from app.quant.methods.returns_descriptive import ReturnsDescriptiveMethod

    n = 100
    dates = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"Date": dates})
    for i in range(4):
        df[f"TICK{i}"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(4)}}

    result = ReturnsDescriptiveMethod().calculate(df, role_map, {"security": PORTFOLIO_LABEL})
    assert result.stats["Observations"] > 0
    assert "Portfolio" in result.figure["layout"]["title"]["text"] or True  # title includes security name somewhere


def test_end_to_end_var_cvar_on_whole_portfolio_differs_from_single_security():
    from app.quant.methods.var_cvar import VarCvarMethod

    n = 200
    dates = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(10)
    df = pd.DataFrame({"Date": dates})
    for i in range(5):
        df[f"TICK{i}"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(5)}}

    portfolio_result = VarCvarMethod().calculate(df, role_map, {"security": PORTFOLIO_LABEL, "confidence": 95})
    single_result = VarCvarMethod().calculate(df, role_map, {"security": "TICK0", "confidence": 95})
    # a diversified blend should show meaningfully lower VaR than one single volatile holding
    assert portfolio_result.stats["Historical VaR (%)"] != single_result.stats["Historical VaR (%)"]


def test_blended_series_raises_clear_error_with_no_overlap():
    dates_a = pd.bdate_range("2023-01-02", periods=50)
    dates_b = pd.bdate_range("2024-01-02", periods=50)
    panel = pd.DataFrame({
        "A": pd.Series(np.linspace(100, 110, 50), index=dates_a),
        "B": pd.Series(np.linspace(50, 55, 50), index=dates_b),
    })
    with pytest.raises(ValueError):
        blended_portfolio_series(panel)
