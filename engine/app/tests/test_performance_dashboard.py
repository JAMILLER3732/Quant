"""
Regression test for a real-data readability bug in PerformanceDashboardMethod:
plotting one legend entry and one always-on scatter text label per security
was fine for a handful of names but became an unreadable overlapping wall of
labels on a real 80+-security portfolio, since Plotly's legend doesn't wrap
without overlapping the plot, and there are only 8 palette colors to begin
with. Past MAX_HIGHLIGHTED securities, only the best/worst-Sharpe names
should get a legend entry / static text label; everything else stays visible
as an unlabeled, muted line (still hoverable, still in the ranking table).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.quant.methods.performance_dashboard import PerformanceDashboardMethod

METHOD = PerformanceDashboardMethod()


def _panel_df(n=120, n_assets=5, seed=13):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    data = {"Date": dates}
    for i in range(n_assets):
        data[f"TICK{i}"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame(data)


def test_small_panel_every_security_gets_a_legend_entry():
    df = _panel_df(n_assets=5)
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(5)}}
    result = METHOD.calculate(df, role_map, {})
    named_traces = [t for t in result.figure["data"] if t.get("showlegend", True) is not False]
    assert len(named_traces) == 5


def test_large_panel_only_highlights_are_named_the_rest_stay_visible_but_muted():
    df = _panel_df(n_assets=20)
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(20)}}
    result = METHOD.calculate(df, role_map, {})

    all_traces = result.figure["data"]
    assert len(all_traces) == 20  # every security still drawn, none dropped

    legend_traces = [t for t in all_traces if t.get("showlegend", True) is not False]
    muted_traces = [t for t in all_traces if t.get("showlegend") is False]
    assert 0 < len(legend_traces) <= 8  # bounded, unlike one-entry-per-security
    assert len(muted_traces) == 20 - len(legend_traces)
    for t in muted_traces:
        assert t["opacity"] < 1.0

    # the full ranking table still has every security regardless of chart highlighting
    assert len(result.tables["ranking"]) == 20
