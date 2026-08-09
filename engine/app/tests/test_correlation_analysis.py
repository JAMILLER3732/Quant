"""
Regression test for a real-data bug found in correlation_analysis: a single
near-zero-variance security (e.g. a money-market fund pegged at $1.00) makes
its correlation with every other security NaN (0/0 in Pearson's formula).
Plain numpy mean/max/min propagate NaN, so the summary stats
(Average/Max/Min Pairwise Correlation) silently came back None for the
*entire* matrix — even across an 82-security real portfolio file where 81 of
82 securities had perfectly valid, computable correlations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.quant.methods.correlation_analysis import CorrelationAnalysisMethod

METHOD = CorrelationAnalysisMethod()


def _panel_df(n=100, n_assets=4, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    data = {"Date": dates}
    for i in range(n_assets):
        data[f"TICK{i}"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame(data)


def test_summary_stats_are_finite_for_normal_multi_asset_panel():
    df = _panel_df()
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(4)}}
    result = METHOD.calculate(df, role_map, {})
    for key in ("Average Pairwise Correlation", "Max Pairwise Correlation", "Min Pairwise Correlation"):
        assert result.stats[key] is not None
        assert -1.0 <= result.stats[key] <= 1.0


def test_zero_variance_security_does_not_null_out_entire_summary():
    # One security pegged flat (a money-market-fund-like ticker) must not
    # poison the summary stats for the other, perfectly valid, pairs.
    df = _panel_df(n_assets=4)
    df["FLAT"] = 1.00  # zero variance throughout
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(4)}, "FLAT": "close"}

    result = METHOD.calculate(df, role_map, {})
    for key in ("Average Pairwise Correlation", "Max Pairwise Correlation", "Min Pairwise Correlation"):
        assert result.stats[key] is not None, f"{key} was null despite 4 other valid securities"
        assert -1.0 <= result.stats[key] <= 1.0

    assert any("FLAT" in w and "zero" in w.lower() for w in result.warnings)

    # The matrix itself should still expose the undefined FLAT correlations as
    # None (never a fabricated number), while every other cell stays numeric.
    flat_row = next(r for r in result.series_csv_rows if r["security"] == "FLAT")
    assert flat_row["TICK0"] is None
    other_row = next(r for r in result.series_csv_rows if r["security"] == "TICK0")
    assert other_row["TICK1"] is not None


def test_large_matrix_drops_cell_text_and_caps_height():
    # A real 30+-security portfolio must not render per-cell numeric labels
    # (unreadable at that density) or balloon to an unusably tall chart.
    df = _panel_df(n_assets=30, seed=11)
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(30)}}
    result = METHOD.calculate(df, role_map, {})
    heatmap_trace = result.figure["data"][0]
    assert heatmap_trace.get("texttemplate") is None
    assert result.figure["layout"]["height"] <= 1400


def test_small_matrix_keeps_cell_text_labels():
    df = _panel_df(n_assets=4)
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(4)}}
    result = METHOD.calculate(df, role_map, {})
    heatmap_trace = result.figure["data"][0]
    assert heatmap_trace.get("texttemplate") == "%{text}"
