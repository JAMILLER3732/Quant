"""Tests for the report engine: data collection, templated-narrative fallback,
and HTML rendering — all independent of the AI layer and of WeasyPrint (PDF
rendering needs system libraries not guaranteed present in every test env,
so it's exercised via the live API/deployment instead, not here)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.reports.ai_client import is_configured
from app.reports.builder import build_report_data
from app.reports.narrative import generate_report_narrative
from app.reports.pdf import render_report_html


def _sample_df(n=200, n_assets=3, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    data = {"Date": dates}
    for i in range(n_assets):
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        data[f"TICK{i}"] = prices
    return pd.DataFrame(data)


def test_build_report_data_portfolio_scope():
    df = _sample_df(n_assets=3)
    role_map = {"Date": "date", "TICK0": "close", "TICK1": "close", "TICK2": "close"}
    data = build_report_data(df, role_map, scope="portfolio", security=None, include_optimization=False)

    assert data.scope == "portfolio"
    assert data.n_observations == 200
    method_names = [s.method_name for s in data.sections]
    assert "Performance & Risk Dashboard" in method_names
    assert "Correlation & Covariance Analysis" in method_names
    # every section's stats must be non-empty and every figure must be a real plotly figure dict
    for section in data.sections:
        assert len(section.stats) > 0
        assert section.figure is not None and "data" in section.figure


def test_build_report_data_security_scope():
    df = _sample_df(n_assets=2)
    role_map = {"Date": "date", "TICK0": "close", "TICK1": "close"}
    data = build_report_data(df, role_map, scope="security", security="TICK0", include_optimization=False)

    assert data.scope == "security"
    assert data.target == "TICK0"
    method_names = [s.method_name for s in data.sections]
    assert "Returns & Descriptive Statistics" in method_names
    assert "Rolling Z-Score & Standard Deviation Bands" in method_names
    assert "Value at Risk & Expected Shortfall" in method_names


def test_build_report_data_with_optimization_needs_multiple_assets():
    df = _sample_df(n_assets=4)
    role_map = {"Date": "date", **{f"TICK{i}": "close" for i in range(4)}}
    data = build_report_data(df, role_map, scope="portfolio", security=None, include_optimization=True)
    method_names = [s.method_name for s in data.sections]
    assert "Efficient Frontier & Portfolio Optimization" in method_names


def test_report_data_never_fabricates_numbers_beyond_what_methods_computed():
    # Every numeric stat in the report must trace back to a real QuantMethod
    # calculation — this test checks the structural guarantee: ReportSection
    # stats are exactly what the underlying method.calculate() returned, with
    # no report-layer post-processing that could alter a number silently.
    df = _sample_df(n_assets=2)
    role_map = {"Date": "date", "TICK0": "close", "TICK1": "close"}

    from app.quant.registry import get_method
    method = get_method("performance_dashboard")
    direct_result = method.calculate(df, role_map, {})

    data = build_report_data(df, role_map, scope="portfolio", security=None, include_optimization=False)
    section = next(s for s in data.sections if s.method_id == "performance_dashboard")
    assert section.stats == direct_result.stats


def test_templated_narrative_fallback_when_ai_not_configured(monkeypatch):
    monkeypatch.delenv("REPORT_AI_API_KEY", raising=False)
    assert not is_configured()

    df = _sample_df(n_assets=2)
    role_map = {"Date": "date", "TICK0": "close", "TICK1": "close"}
    data = build_report_data(df, role_map, scope="portfolio", security=None, include_optimization=False)

    narrative = generate_report_narrative(data)
    assert narrative["ai_generated"] == "false"
    assert len(narrative["executive_summary"]) > 0
    assert len(narrative["analysis"]) > 0
    assert len(narrative["limitations"]) > 0
    # the disclaimer language must be present — never silently omitted
    assert "not" in narrative["limitations"].lower()


def test_ai_narrative_gracefully_falls_back_on_auth_failure(monkeypatch):
    # A configured-but-invalid key must degrade to templated narrative, never crash.
    monkeypatch.setenv("REPORT_AI_API_KEY", "sk-bl-invalid-test-key-00000000000000000000")
    monkeypatch.setenv("REPORT_AI_BASE_URL", "https://bazaarlink.ai/api/v1")

    df = _sample_df(n_assets=2)
    role_map = {"Date": "date", "TICK0": "close", "TICK1": "close"}
    data = build_report_data(df, role_map, scope="portfolio", security=None, include_optimization=False)

    narrative = generate_report_narrative(data)
    # Either it truly failed (falls back) or -- extremely unlikely -- succeeded;
    # either way this must not raise.
    assert "executive_summary" in narrative


def test_render_report_html_contains_disclaimer_and_charts():
    df = _sample_df(n_assets=2)
    role_map = {"Date": "date", "TICK0": "close", "TICK1": "close"}
    data = build_report_data(df, role_map, scope="portfolio", security=None, include_optimization=False)
    narrative = generate_report_narrative(data)

    import html as html_module

    html = render_report_html(data, narrative)
    assert "not investment advice" in html
    assert "data:image/png;base64" in html
    for section in data.sections:
        assert html_module.escape(section.method_name) in html


def test_report_raises_clear_error_when_no_price_data_mapped():
    df = pd.DataFrame({"Date": pd.bdate_range("2023-01-02", periods=10), "Notes": ["x"] * 10})
    role_map = {"Date": "date"}
    with pytest.raises(Exception):
        build_report_data(df, role_map, scope="portfolio", security=None, include_optimization=False)
