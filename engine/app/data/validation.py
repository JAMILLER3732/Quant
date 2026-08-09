"""
Data-quality validation pipeline.

Runs a battery of structural and financial-data checks on a mapped dataset
and produces a human-readable Data Quality Report. Nothing here blocks
calculation outright (that's decided by the requesting quant method's
`validate` step) — this module only observes and reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}


@dataclass
class Issue:
    severity: str  # "info" | "warning" | "error"
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    n_rows: int
    n_columns: int
    n_securities: int
    date_range: tuple[str, str] | None
    inferred_frequency: str | None
    issues: list[Issue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_securities": self.n_securities,
            "date_range": self.date_range,
            "inferred_frequency": self.inferred_frequency,
            "issues": [
                {"severity": i.severity, "code": i.code, "message": i.message, "detail": i.detail}
                for i in sorted(self.issues, key=lambda x: -SEVERITY_ORDER[x.severity])
            ],
            "has_blocking_errors": any(i.severity == "error" for i in self.issues),
        }


def _infer_frequency(dates: pd.Series) -> str | None:
    d = pd.to_datetime(dates.dropna().drop_duplicates().sort_values())
    if len(d) < 3:
        return None
    deltas = d.diff().dropna().dt.days
    if deltas.empty:
        return None
    median_gap = deltas.median()
    if median_gap <= 1:
        return "daily"
    if 2 <= median_gap <= 4:
        return "daily (business days)"
    if 5 <= median_gap <= 9:
        return "weekly"
    if 25 <= median_gap <= 35:
        return "monthly"
    if 85 <= median_gap <= 100:
        return "quarterly"
    if 360 <= median_gap <= 370:
        return "annual"
    return f"irregular (median gap ~{median_gap:.0f} days)"


def run_validation(
    df: pd.DataFrame,
    role_map: dict[str, str],
    price_like_roles: tuple[str, ...] = ("open", "high", "low", "close", "adj_close"),
) -> QualityReport:
    """
    df: the raw uploaded dataframe (post basic cleanup)
    role_map: {column_name: role} as confirmed/edited by the user
    """
    issues: list[Issue] = []
    col_by_role: dict[str, list[str]] = {}
    for col, role in role_map.items():
        col_by_role.setdefault(role, []).append(col)

    n_rows, n_cols = df.shape
    date_cols = col_by_role.get("date", [])
    ticker_cols = col_by_role.get("ticker", [])

    n_securities = 1
    if ticker_cols:
        n_securities = int(df[ticker_cols[0]].nunique(dropna=True))
    else:
        # wide format: each numeric non-role (or explicitly price-mapped) column is a security
        price_cols = [c for c in df.columns if role_map.get(c) in ("close", "adj_close", "returns")]
        if len(price_cols) > 1:
            n_securities = len(price_cols)

    date_range: tuple[str, str] | None = None
    inferred_freq: str | None = None

    if not date_cols:
        issues.append(Issue("error", "no_date_column",
                             "No column was mapped to Date. A date/time index is required for virtually every "
                             "time-series calculation (returns, volatility, drawdown, etc.)."))
    else:
        date_col = date_cols[0]
        parsed = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        n_invalid = int(parsed.isna().sum() - df[date_col].isna().sum())
        if n_invalid > 0:
            issues.append(Issue("error", "invalid_dates",
                                 f"{n_invalid} value(s) in '{date_col}' could not be parsed as dates.",
                                 {"column": date_col, "count": n_invalid}))
        valid = parsed.dropna()
        if not valid.empty:
            date_range = (str(valid.min().date()), str(valid.max().date()))
            inferred_freq = _infer_frequency(valid)

        # duplicate dates (within same ticker if long-format, else outright)
        if ticker_cols:
            dup_mask = df.duplicated(subset=[date_col, ticker_cols[0]], keep=False)
        else:
            dup_mask = df.duplicated(subset=[date_col], keep=False)
        n_dup = int(dup_mask.sum())
        if n_dup > 0:
            issues.append(Issue("warning", "duplicate_dates",
                                 f"{n_dup} duplicate date observation(s) found"
                                 + (" per security." if ticker_cols else "."),
                                 {"count": n_dup}))

        # gaps relative to inferred frequency (business-day heuristic only)
        if inferred_freq and "daily" in inferred_freq and not valid.empty:
            full_range = pd.bdate_range(valid.min(), valid.max())
            missing = full_range.difference(pd.DatetimeIndex(valid.unique()))
            pct_missing = len(missing) / max(len(full_range), 1)
            if pct_missing > 0.05:
                issues.append(Issue("warning", "date_gaps",
                                     f"~{pct_missing*100:.1f}% of expected business days are absent from the "
                                     "series — check for missing trading days or a sparse upload.",
                                     {"missing_business_days": int(len(missing))}))

    # missing values on numeric/price-like columns
    numeric_role_cols = [c for c, r in role_map.items() if r in price_like_roles + ("returns", "weight", "quantity", "pnl", "volume")]
    total_cells = 0
    total_missing = 0
    for col in numeric_role_cols:
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_non_numeric = int(coerced.isna().sum() - df[col].isna().sum())
        n_missing = int(df[col].isna().sum())
        total_cells += len(df)
        total_missing += n_missing
        if n_non_numeric > 0:
            issues.append(Issue("error", "non_numeric_values",
                                 f"'{col}' contains {n_non_numeric} value(s) that are not numeric.",
                                 {"column": col, "count": n_non_numeric}))
        if n_missing > 0:
            pct = n_missing / len(df) * 100
            sev = "error" if pct > 20 else "warning"
            issues.append(Issue(sev, "missing_values",
                                 f"'{col}' has {n_missing} missing value(s) ({pct:.1f}%).",
                                 {"column": col, "count": n_missing, "pct": round(pct, 2)}))

        # zero/negative prices where inappropriate
        if role_map.get(col) in ("open", "high", "low", "close", "adj_close"):
            n_nonpositive = int((coerced <= 0).sum())
            if n_nonpositive > 0:
                issues.append(Issue("error", "nonpositive_prices",
                                     f"'{col}' has {n_nonpositive} zero/negative price value(s), which is invalid "
                                     "for return calculations (log/percent returns are undefined or nonsensical).",
                                     {"column": col, "count": n_nonpositive}))

            # extreme single-period jumps (outlier heuristic, not itself an error)
            clean = coerced.dropna()
            if len(clean) > 5:
                pct_change = clean.pct_change().dropna()
                extreme = pct_change[pct_change.abs() > 0.5]
                if not extreme.empty:
                    issues.append(Issue("warning", "extreme_price_moves",
                                         f"'{col}' has {len(extreme)} single-period move(s) greater than 50%, "
                                         "which may indicate a stock split, data error, or genuine extreme event. "
                                         "Verify whether prices are split/dividend adjusted.",
                                         {"column": col, "count": int(len(extreme))}))

    if numeric_role_cols and total_cells > 0:
        overall_pct = total_missing / total_cells * 100
        if overall_pct > 0:
            issues.append(Issue("info", "overall_missing_summary",
                                 f"Dataset-wide: {overall_pct:.1f}% of numeric observations are missing.",
                                 {"pct": round(overall_pct, 2)}))

    # adjusted vs unadjusted close guidance
    if "close" in col_by_role and "adj_close" not in col_by_role:
        issues.append(Issue("info", "unadjusted_close_only",
                             "Only an unadjusted Close price is mapped. If this security had dividends or "
                             "splits during the sample period, return calculations will be biased. Provide "
                             "Adjusted Close if available."))

    # ticker-level start/end alignment
    if ticker_cols and date_cols:
        tcol, dcol = ticker_cols[0], date_cols[0]
        parsed_dates = pd.to_datetime(df[dcol], errors="coerce", format="mixed")
        tmp = pd.DataFrame({"t": df[tcol], "d": parsed_dates}).dropna()
        if not tmp.empty:
            spans = tmp.groupby("t")["d"].agg(["min", "max", "count"])
            if spans["min"].nunique() > 1 or spans["max"].nunique() > 1:
                issues.append(Issue("warning", "misaligned_start_end",
                                     "Securities in this dataset have different start/end dates. Cross-sectional "
                                     "calculations (correlation, portfolio optimization) will only use the "
                                     "overlapping window unless you restrict the date range.",
                                     {"spans": {str(k): {"start": str(v["min"].date()), "end": str(v["max"].date()),
                                                          "n_obs": int(v["count"])}
                                                for k, v in spans.iterrows()}}))
            short = spans[spans["count"] < 30]
            if not short.empty:
                issues.append(Issue("warning", "insufficient_history",
                                     f"{len(short)} security(ies) have fewer than 30 observations, which is too "
                                     "little for most statistical estimates (volatility, correlation, regression) "
                                     "to be reliable.",
                                     {"securities": list(short.index.astype(str))}))

    if n_rows < 30:
        issues.append(Issue("warning", "short_dataset",
                             f"Only {n_rows} rows total — many quant methods (rolling stats, GARCH, factor "
                             "regressions) need substantially more observations to produce stable estimates."))

    if not issues:
        issues.append(Issue("info", "clean", "No structural data-quality issues were detected."))

    return QualityReport(
        n_rows=n_rows,
        n_columns=n_cols,
        n_securities=n_securities,
        date_range=date_range,
        inferred_frequency=inferred_freq,
        issues=issues,
    )
