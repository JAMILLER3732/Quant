"""
Report data collection: runs a curated set of real QuantMethod calculations
against a dataset and assembles the results into a single structure the
template/PDF renderer and the AI narrative layer both consume.

Every number in the resulting ReportData came from an actual QuantMethod
calculation — this module does no computation of its own beyond selecting
which methods to run and packaging their output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.data.reshape import price_panel
from app.quant.registry import get_method


@dataclass
class ReportSection:
    method_id: str
    method_name: str
    stats: dict[str, Any]
    figure: dict[str, Any] | None
    tables: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReportData:
    title: str
    subtitle: str
    scope: str  # "security" | "portfolio"
    target: str  # security ticker, or "Portfolio" / comma-joined list
    date_range: tuple[str, str] | None
    n_observations: int
    sections: list[ReportSection]
    warnings: list[str]


def build_report_data(
    df: pd.DataFrame,
    role_map: dict[str, str],
    scope: str,
    security: str | None,
    include_optimization: bool,
) -> ReportData:
    panel, price_role = price_panel(df, role_map)
    all_warnings: list[str] = []
    if price_role == "close":
        all_warnings.append("Report uses unadjusted Close prices — figures may be distorted around dividends/splits.")

    if scope == "security":
        target = security if security in panel.columns else panel.columns[0]
        securities = [target]
        title = f"{target} — Equity Research Note"
        subtitle = "Single-security performance, risk, and statistical analysis"
    else:
        target = "Portfolio"
        securities = list(panel.columns)
        title = "Portfolio Analysis Report"
        subtitle = f"{len(securities)} securities — performance, risk, and diversification analysis"

    prices = panel[securities].dropna(how="all")
    date_range = (str(prices.index.min().date()), str(prices.index.max().date())) if len(prices) else None

    sections: list[ReportSection] = []

    def run(method_id: str, params: dict[str, Any]) -> None:
        method = get_method(method_id)
        check = method.check_requirements(role_map, df)
        if not check.satisfied:
            return
        try:
            result = method.calculate(df, role_map, params)
        except ValueError:
            return
        sections.append(ReportSection(
            method_id=method_id, method_name=method.name, stats=result.stats,
            figure=result.figure, tables=result.tables, warnings=result.warnings,
        ))
        all_warnings.extend(result.warnings)

    if scope == "security":
        run("returns_descriptive", {"security": target, "return_type": "simple", "rolling_window": 21})
        run("rolling_zscore", {"security": target, "window": 20, "band_k": 2.0})
        run("var_cvar", {"security": target, "confidence": 95})
    else:
        run("performance_dashboard", {"risk_free_rate": 0.0, "var_confidence": 95})
        run("correlation_analysis", {"method": "pearson", "rolling_window": 60})
        if include_optimization and len(securities) >= 2:
            run("efficient_frontier", {"risk_free_rate": 0.0, "n_random_portfolios": 3000})

    return ReportData(
        title=title, subtitle=subtitle, scope=scope, target=target,
        date_range=date_range, n_observations=len(prices), sections=sections,
        warnings=sorted(set(all_warnings)),
    )
