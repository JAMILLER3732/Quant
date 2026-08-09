"""
Regression test for a real-data crash in HrpAllocationMethod: a single
near-zero-variance security (e.g. a money-market fund pegged at $1.00) makes
its correlation-distance to every other security NaN/Inf, which scipy's
linkage() rejects outright ("condensed distance matrix must contain only
finite values") — crashing HRP entirely, even when every other security in
an 82-security real portfolio was perfectly clusterable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.quant.methods.hrp_allocation import HrpAllocationMethod

METHOD = HrpAllocationMethod()


def _panel_df(n=100, n_assets=5, seed=9):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    data = {"Date": dates}
    for i in range(n_assets):
        data[f"TICK{i}"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame(data)


def test_zero_variance_security_is_excluded_not_crashing():
    df = _panel_df(n_assets=5)
    df["FLAT"] = 1.00
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(5)}, "FLAT": "close"}

    result = METHOD.calculate(df, role_map, {})  # must not raise
    assert "FLAT" not in result.stats["Securities"]
    assert any("FLAT" in w and "zero" in w.lower() for w in result.warnings)
    weight_secs = {row["security"] for row in result.series_csv_rows}
    assert "FLAT" not in weight_secs
    assert weight_secs == {f"TICK{i}" for i in range(5)}


def test_normal_panel_unaffected():
    df = _panel_df(n_assets=5)
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(5)}}
    result = METHOD.calculate(df, role_map, {})
    assert not any("zero variance" in w.lower() for w in result.warnings)
    assert len(result.series_csv_rows) == 5
