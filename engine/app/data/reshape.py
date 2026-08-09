"""
Canonical reshaping: turn a raw (df, role_map) pair into a tidy wide panel
indexed by date with one column per security, regardless of whether the
source was long-format OHLCV, wide prices, or a returns table.

This is the single place that understands "Format A/B/C/D/E" from the
product spec, so every QuantMethod can just ask for a price or returns
panel without re-deriving pivot logic.
"""
from __future__ import annotations

import pandas as pd


class ReshapeError(Exception):
    pass


def _date_index(df: pd.DataFrame, date_col: str) -> pd.Series:
    parsed = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    return parsed


def extract_panel(
    df: pd.DataFrame,
    role_map: dict[str, str],
    value_role: str,
) -> pd.DataFrame:
    """
    Returns a wide DataFrame: index=Date (sorted, deduped), columns=security
    identifiers, values=float. `value_role` is e.g. "close", "adj_close",
    "returns", "portfolio_value".
    """
    date_cols = [c for c, r in role_map.items() if r == "date"]
    if not date_cols:
        raise ReshapeError("No column is mapped to 'date'.")
    date_col = date_cols[0]

    ticker_cols = [c for c, r in role_map.items() if r == "ticker"]
    value_cols = [c for c, r in role_map.items() if r == value_role]

    if not value_cols:
        raise ReshapeError(f"No column is mapped to '{value_role}'.")

    dates = _date_index(df, date_col)
    work = df.copy()
    work["_date"] = dates
    work = work.dropna(subset=["_date"])

    if ticker_cols:
        # long format: Date + Ticker + value column
        tcol = ticker_cols[0]
        vcol = value_cols[0]
        work["_value"] = pd.to_numeric(work[vcol], errors="coerce")
        pivot = work.pivot_table(index="_date", columns=tcol, values="_value", aggfunc="last")
    else:
        # wide format: one column per security already
        work_indexed = work.set_index("_date")
        cols = {vc: pd.to_numeric(work_indexed[vc], errors="coerce") for vc in value_cols}
        pivot = pd.DataFrame(cols)
        pivot = pivot.groupby(level=0).last()

    pivot = pivot.sort_index()
    pivot.index.name = "date"
    return pivot


def extract_series(df: pd.DataFrame, role_map: dict[str, str], role: str) -> pd.Series:
    """Convenience for single-series roles (e.g. benchmark, risk_free, portfolio_value)."""
    panel = extract_panel(df, role_map, role)
    return panel.iloc[:, 0]


def price_panel(df: pd.DataFrame, role_map: dict[str, str]) -> tuple[pd.DataFrame, str]:
    """Prefer adjusted close over close if both are mapped. Returns (panel, role_used)."""
    roles_present = set(role_map.values())
    if "adj_close" in roles_present:
        return extract_panel(df, role_map, "adj_close"), "adj_close"
    if "close" in roles_present:
        return extract_panel(df, role_map, "close"), "close"
    raise ReshapeError("Neither 'close' nor 'adj_close' is mapped — cannot build a price panel.")


def returns_panel(df: pd.DataFrame, role_map: dict[str, str]) -> pd.DataFrame:
    """Use a directly-mapped 'returns' role if present, else derive simple returns from prices."""
    roles_present = set(role_map.values())
    if "returns" in roles_present:
        return extract_panel(df, role_map, "returns")
    panel, _ = price_panel(df, role_map)
    return panel.pct_change().dropna(how="all")
