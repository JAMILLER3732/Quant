"""
Regression tests for bugs found by testing column detection against real
uploaded Excel data (a Bloomberg-style price-history export). Three distinct
bugs surfaced in one file and are pinned here so they can't silently return:

1. A raw numeric column (e.g. an index level like 8647.569) was being handed
   to pd.to_datetime, which silently reinterprets bare numbers as
   nanoseconds-since-epoch and "succeeds" on virtually any numeric column —
   misclassifying a benchmark/price column as a date column.
2. The 'close' role's name pattern included 'px_last', which collided with a
   real Bloomberg export convention of labeling the date axis column itself
   'PX_LAST' (the field name of the query, not 'Date') — actively pulling the
   real date column into the wrong role.
3. A near-empty "phantom used-range" artifact column (Excel padding with a
   handful of stray non-breaking-space cells) was confidently classified as
   a ticker column by the categorical-string heuristic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.column_detection import guess_columns, guess_structure
from app.data.ingestion import ingest_file


def test_numeric_index_column_is_not_misdetected_as_date():
    # A benchmark/index level column — must never be flagged as a date purely
    # because pd.to_datetime can coerce any float into a nanosecond timestamp.
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=20),
        "SPTR Index": np.linspace(8500, 8700, 20),
    })
    guesses = {g.column: g.role for g in guess_columns(df)}
    assert guesses["SPTR Index"] != "date"


def test_date_column_labeled_with_field_name_is_still_detected_by_content():
    # Real Bloomberg exports sometimes label the date axis 'PX_LAST' (the
    # query field name) instead of 'Date'. Content-based detection must catch
    # it, and the removed 'px_last' name-pattern must not misroute it as 'close'.
    df = pd.DataFrame({
        "PX_LAST": pd.date_range("2024-01-01", periods=20),
        "AAPL": np.linspace(150, 160, 20),
    })
    guesses = {g.column: g.role for g in guess_columns(df)}
    assert guesses["PX_LAST"] == "date"


def test_near_empty_artifact_column_is_ignored_not_misclassified():
    # A "phantom used-range" column: 1200 nulls, 7 stray whitespace cells.
    n = 1200
    values = [None] * n
    for i in range(7):
        values[i * 100] = "\xa0"
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n),
        "AAPL": np.linspace(150, 200, n),
        "Unnamed: 85": values,
    })
    guesses = {g.column: g.role for g in guess_columns(df)}
    assert guesses["Unnamed: 85"] == "ignore"


def test_whitespace_only_artifact_column_is_dropped_at_ingestion():
    import io

    n = 50
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n).astype(str),
        "AAPL": np.linspace(150, 160, n),
    })
    df["Unnamed: 5"] = [None] * n
    df.loc[3, "Unnamed: 5"] = "\xa0"

    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    result = ingest_file("test.xlsx", buf.getvalue())
    assert "Unnamed: 5" not in result.df.columns


def test_full_wide_price_structure_detected_end_to_end():
    # The combined regression: with all three bugs fixed, a realistic wide
    # price panel (date column with a non-standard header + numeric index
    # column + a junk artifact column) should still resolve to wide_prices.
    n = 30
    df = pd.DataFrame({
        "PX_LAST": pd.date_range("2024-01-01", periods=n),
        "SPTR Index": np.linspace(8500, 8700, n),
        "AAPL": np.linspace(150, 160, n),
        "Unnamed: 9": [None] * n,
    })
    guesses = guess_columns(df)
    structure = guess_structure(df, guesses)
    assert structure.format_id == "wide_prices"


def test_wide_multi_security_file_auto_maps_every_ticker_to_close():
    # Regression for a real 82-security portfolio export: a Date column plus
    # dozens of bare ticker-symbol headers with no "Close"/"Price" in the name
    # at all. Without structure-level fallback, every one of those columns
    # falls back to role "ignore" and the user has to hand-map 80+ dropdowns
    # before running a single calculation. They must instead default to
    # "close" so the file is immediately usable, with a lower confidence score
    # (0.55, below the 0.9 a real name/content match gets) so the UI can still
    # visibly flag them as "please verify" rather than claiming certainty.
    n = 60
    tickers = [f"TICK{i}" for i in range(82)]
    data = {"Date": pd.bdate_range("2024-01-02", periods=n)}
    rng = np.random.default_rng(1)
    for t in tickers:
        data[t] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    df = pd.DataFrame(data)

    guesses = guess_columns(df)
    roles = {g.column: g.role for g in guesses}
    assert all(roles[t] == "close" for t in tickers)
    assert all(g.confidence == 0.55 for g in guesses if g.column in tickers)

    structure = guess_structure(df, guesses)
    assert structure.format_id == "wide_prices"


def test_wide_format_fallback_does_not_apply_when_ticker_column_present():
    # A long/stacked format (Date + Ticker + Close) must NOT trigger the wide
    # fallback — there, unlabelled numeric columns are a genuinely different
    # shape (e.g. Volume) and should stay "ignore" rather than being guessed.
    n = 40
    df = pd.DataFrame({
        "Date": list(pd.bdate_range("2024-01-02", periods=n // 2)) * 2,
        "Ticker": ["AAPL"] * (n // 2) + ["MSFT"] * (n // 2),
        "Close": np.linspace(150, 160, n),
        "SomeOtherNumericColumn": np.linspace(1, 2, n),
    })
    guesses = {g.column: g.role for g in guess_columns(df)}
    assert guesses["SomeOtherNumericColumn"] == "ignore"
