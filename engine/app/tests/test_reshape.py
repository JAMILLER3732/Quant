from __future__ import annotations

import pandas as pd

from app.data.reshape import extract_panel, price_panel


def test_wide_price_panel_aligns_dates_correctly():
    """Regression test: a prior bug misaligned wide-format columns against the
    date index (pd.DataFrame(dict_of_series, index=...) re-indexes each Series
    by its own original index rather than positionally), producing all-NaN panels."""
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "AAPL": [100.0, 101.0, 102.0, 103.0, 104.0],
        "MSFT": [200.0, 199.0, 201.0, 202.0, 203.0],
    })
    role_map = {"Date": "date", "AAPL": "close", "MSFT": "close"}
    panel, role_used = price_panel(df, role_map)

    assert role_used == "close"
    assert list(panel.columns) == ["AAPL", "MSFT"]
    assert panel.isna().sum().sum() == 0
    assert panel["AAPL"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert panel["MSFT"].tolist() == [200.0, 199.0, 201.0, 202.0, 203.0]


def test_long_format_ohlcv_pivots_by_ticker():
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "Ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
        "Close": [100.0, 200.0, 101.0, 199.0],
    })
    role_map = {"Date": "date", "Ticker": "ticker", "Close": "close"}
    panel = extract_panel(df, role_map, "close")
    assert set(panel.columns) == {"AAPL", "MSFT"}
    assert panel.loc[pd.Timestamp("2024-01-01"), "AAPL"] == 100.0
    assert panel.loc[pd.Timestamp("2024-01-02"), "MSFT"] == 199.0
