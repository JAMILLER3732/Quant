"""
Regression tests for StressTestingMethod:
1. A beta-scaled historical shock can imply a security loses more than 100%
   of its value (shock_i = beta_i * benchmark_shock, and beta_i can exceed
   100/|benchmark_shock| for a high-beta name) — which produced a NEGATIVE
   "Shocked Price" in real-portfolio testing (a stock literally cannot have
   a negative price). Shocks must be floored at -100%, with a warning.
2. Past ~20 securities, drawing one bar per security produced an unreadable
   wall of overlapping x-axis tick labels — the chart must fall back to only
   the best/worst N, with every security's numbers still in the full table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.quant.methods.stress_testing import StressTestingMethod

METHOD = StressTestingMethod()


def _panel_with_benchmark(n=300, n_assets=3, seed=21):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    data = {"Date": dates}
    # A high-volatility security whose beta to the benchmark will be large.
    bench = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    data["BENCH"] = bench
    bench_returns = pd.Series(bench).pct_change().fillna(0).values
    # HIGHBETA moves ~4x the benchmark every day (plus noise) -> beta >> 1.
    highbeta_returns = 4.0 * bench_returns + rng.normal(0, 0.002, n)
    data["HIGHBETA"] = 100 * np.cumprod(1 + highbeta_returns)
    for i in range(n_assets - 1):
        data[f"TICK{i}"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame(data)


def test_historical_scenario_never_produces_a_negative_shocked_price():
    df = _panel_with_benchmark()
    role_map = {"Date": "date", "BENCH": "benchmark", "HIGHBETA": "close",
                **{f"TICK{i}": "close" for i in range(2)}}
    result = METHOD.calculate(df, role_map, {"scenario": "gfc_2008"})
    for row in result.series_csv_rows:
        assert row["Shocked Price"] is not None
        assert row["Shocked Price"] >= 0, f"{row['Security']} shocked price went negative: {row['Shocked Price']}"
        assert row["Shock (%)"] >= -100.0


def test_clipped_security_is_named_in_a_warning():
    df = _panel_with_benchmark()
    role_map = {"Date": "date", "BENCH": "benchmark", "HIGHBETA": "close",
                **{f"TICK{i}": "close" for i in range(2)}}
    result = METHOD.calculate(df, role_map, {"scenario": "gfc_2008"})
    assert any("HIGHBETA" in w and "100%" in w for w in result.warnings)


def test_custom_shock_within_bounds_is_never_clipped():
    df = _panel_with_benchmark()
    role_map = {"Date": "date", "BENCH": "benchmark", "HIGHBETA": "close",
                **{f"TICK{i}": "close" for i in range(2)}}
    result = METHOD.calculate(df, role_map, {"scenario": "custom_uniform", "shock_pct": -20})
    assert not any("100%" in w for w in result.warnings)
    for row in result.series_csv_rows:
        assert row["Shock (%)"] == -20.0


def test_many_securities_chart_shows_only_best_worst_but_table_has_everyone():
    rng = np.random.default_rng(5)
    n, n_assets = 100, 30
    dates = pd.bdate_range("2023-01-02", periods=n)
    data = {"Date": dates}
    for i in range(n_assets):
        data[f"TICK{i}"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    df = pd.DataFrame(data)
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(n_assets)}}

    result = METHOD.calculate(df, role_map, {"scenario": "custom_uniform", "shock_pct": -15})
    assert len(result.series_csv_rows) == n_assets  # full table
    assert len(result.figure["data"][0]["x"]) <= 20  # chart capped
    assert "best/worst" in result.figure["layout"]["title"]["text"]
