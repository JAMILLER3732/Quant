"""
End-to-end API tests: upload -> map columns -> check requirements -> calculate,
for every registered method. Guards against serialization bugs (numpy/datetime
in JSON responses) and pivot/reshape bugs that unit tests on individual
functions can miss.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.quant.registry import REGISTRY

client = TestClient(app)


def _sample_csv_bytes() -> bytes:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=300)

    def gbm(s0, mu, sigma, n):
        dt = 1 / 252
        shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
        return s0 * np.exp(np.cumsum(shocks))

    df = pd.DataFrame({
        "Date": dates,
        "AAPL": gbm(150, 0.12, 0.28, len(dates)),
        "MSFT": gbm(300, 0.10, 0.25, len(dates)),
        "SPY": gbm(430, 0.08, 0.16, len(dates)),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture()
def mapped_dataset_id() -> str:
    content = _sample_csv_bytes()
    resp = client.post("/api/upload", files={"file": ("sample.csv", content, "text/csv")})
    assert resp.status_code == 200, resp.text
    dataset_id = resp.json()["dataset_id"]

    mapping_resp = client.post(
        f"/api/datasets/{dataset_id}/mapping",
        json={"role_map": {"Date": "date", "AAPL": "close", "MSFT": "close", "SPY": "close"}},
    )
    assert mapping_resp.status_code == 200, mapping_resp.text
    return dataset_id


@pytest.fixture()
def mapped_dataset_id_with_benchmark() -> str:
    """Same as mapped_dataset_id, but SPY is mapped to 'benchmark' instead of
    'close', for methods that regress a security against a market factor."""
    content = _sample_csv_bytes()
    resp = client.post("/api/upload", files={"file": ("sample.csv", content, "text/csv")})
    dataset_id = resp.json()["dataset_id"]
    mapping_resp = client.post(
        f"/api/datasets/{dataset_id}/mapping",
        json={"role_map": {"Date": "date", "AAPL": "close", "MSFT": "close", "SPY": "benchmark"}},
    )
    assert mapping_resp.status_code == 200, mapping_resp.text
    return dataset_id


# factor_analysis needs a mapped benchmark (see mapped_dataset_id_with_benchmark below)
# rather than the generic all-close mapping every other method is fine with.
METHODS_NEEDING_GENERIC_MAPPING = [m for m in REGISTRY.keys() if m != "factor_analysis"]


def test_upload_returns_structure_and_quality_report():
    content = _sample_csv_bytes()
    resp = client.post("/api/upload", files={"file": ("sample.csv", content, "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == 300
    assert body["structure_guess"]["format_id"] == "wide_prices"
    assert "quality_report" in body


def test_upload_rejects_empty_file():
    resp = client.post("/api/upload", files={"file": ("empty.csv", b"", "text/csv")})
    assert resp.status_code == 400


def test_upload_rejects_unsupported_extension():
    resp = client.post("/api/upload", files={"file": ("data.txt", b"a,b\n1,2", "text/plain")})
    assert resp.status_code == 400


def test_methods_endpoint_lists_all_registered_methods():
    resp = client.get("/api/methods")
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["methods"]}
    assert ids == set(REGISTRY.keys())


def test_requirements_satisfied_after_mapping(mapped_dataset_id):
    resp = client.get(f"/api/datasets/{mapped_dataset_id}/methods/returns_descriptive/requirements")
    assert resp.status_code == 200
    body = resp.json()
    assert body["satisfied"] is True
    assert set(body["dynamic_param_options"]["security"]) == {"AAPL", "MSFT", "SPY"}


def test_requirements_satisfied_by_wide_format_auto_mapping():
    # A real multi-security file is typically "wide" — Date + one numeric column
    # per ticker, with no "Close"/"Price" in the header names at all (just the
    # tickers themselves). guess_columns() auto-resolves those to role "close"
    # from the sheet's overall shape, so requirements should already be
    # satisfied before the user manually maps anything — otherwise a real
    # 80+-security upload would require remapping every column by hand.
    content = _sample_csv_bytes()
    resp = client.post("/api/upload", files={"file": ("sample.csv", content, "text/csv")})
    dataset_id = resp.json()["dataset_id"]
    assert resp.json()["role_map"]["AAPL"] == "close"
    assert resp.json()["role_map"]["MSFT"] == "close"
    req = client.get(f"/api/datasets/{dataset_id}/methods/returns_descriptive/requirements")
    assert req.status_code == 200
    assert req.json()["satisfied"] is True


def test_requirements_not_satisfied_with_no_price_columns_at_all():
    df = pd.DataFrame({"Date": pd.bdate_range("2022-01-03", periods=50), "Notes": ["x"] * 50})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    resp = client.post("/api/upload", files={"file": ("notes.csv", buf.getvalue(), "text/csv")})
    dataset_id = resp.json()["dataset_id"]
    req = client.get(f"/api/datasets/{dataset_id}/methods/returns_descriptive/requirements")
    assert req.status_code == 200
    assert req.json()["satisfied"] is False


@pytest.mark.parametrize("method_id", METHODS_NEEDING_GENERIC_MAPPING)
def test_calculate_every_registered_method_returns_valid_json(mapped_dataset_id, method_id):
    resp = client.post(
        f"/api/datasets/{mapped_dataset_id}/calculate/{method_id}",
        json={"params": {"security": "AAPL"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "figure" in body and "data" in body["figure"]
    assert len(body["figure"]["data"]) > 0
    assert isinstance(body["stats"], dict) and len(body["stats"]) > 0
    assert isinstance(body["csv_rows"], list) and len(body["csv_rows"]) > 0
    # every x/y value in every trace must already be JSON-safe scalars/strings
    for trace in body["figure"]["data"]:
        for axis in ("x", "y"):
            if axis in trace and trace[axis]:
                assert isinstance(trace[axis][0], (str, int, float, type(None)))


def test_factor_analysis_requires_benchmark(mapped_dataset_id):
    # mapped_dataset_id maps SPY to 'close', not 'benchmark' -> requirements should fail cleanly.
    req = client.get(f"/api/datasets/{mapped_dataset_id}/methods/factor_analysis/requirements")
    assert req.status_code == 200
    assert req.json()["satisfied"] is False


def test_factor_analysis_calculates_with_mapped_benchmark(mapped_dataset_id_with_benchmark):
    resp = client.post(
        f"/api/datasets/{mapped_dataset_id_with_benchmark}/calculate/factor_analysis",
        json={"params": {"security": "AAPL"}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "Beta" in body["stats"]
    assert "R-squared" in body["stats"]


def test_calculate_unknown_method_404(mapped_dataset_id):
    resp = client.post(f"/api/datasets/{mapped_dataset_id}/calculate/not_a_real_method", json={"params": {}})
    assert resp.status_code == 404


def test_calculate_with_bad_dataset_id_404():
    resp = client.post("/api/datasets/nonexistent-id/calculate/returns_descriptive", json={"params": {}})
    assert resp.status_code == 404


def test_ewma_requires_fast_span_less_than_slow_span(mapped_dataset_id):
    resp = client.post(
        f"/api/datasets/{mapped_dataset_id}/calculate/ewma_crossover",
        json={"params": {"security": "AAPL", "fast_span": 50, "slow_span": 10}},
    )
    assert resp.status_code == 422
    assert "must be smaller than" in resp.json()["detail"]["message"]
