"""HRP allocation tests: weights sum to 1, are non-negative, and a
degenerate/independent-asset case reduces to inverse-variance weighting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.quant.portfolio import hrp_weights


def test_hrp_weights_sum_to_one_and_nonnegative():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-01", periods=300)
    data = rng.normal(0.0005, 0.01, size=(300, 5))
    returns = pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D", "E"])

    weights = hrp_weights(returns)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()
    assert set(weights.index) == {"A", "B", "C", "D", "E"}


def test_hrp_two_independent_assets_matches_inverse_variance():
    # With exactly 2 uncorrelated assets, HRP's recursive bisection reduces to
    # simple inverse-variance weighting between them.
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2021-01-01", periods=1000)
    low_vol = rng.normal(0.0003, 0.005, 1000)
    high_vol = rng.normal(0.0003, 0.02, 1000)
    returns = pd.DataFrame({"LOW": low_vol, "HIGH": high_vol}, index=dates)

    weights = hrp_weights(returns)
    var_low, var_high = returns["LOW"].var(), returns["HIGH"].var()
    expected_low = (1 / var_low) / (1 / var_low + 1 / var_high)

    assert weights["LOW"] == pytest.approx(expected_low, abs=0.02)
    assert weights["LOW"] > weights["HIGH"]  # lower-volatility asset gets more weight


def test_hrp_assigns_less_weight_to_highly_correlated_redundant_asset():
    # A near-duplicate of an existing asset should not double that cluster's
    # aggregate weight versus a diversifying independent asset.
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2021-01-01", periods=500)
    base = rng.normal(0.0004, 0.01, 500)
    duplicate = base + rng.normal(0, 0.0005, 500)  # near-identical to `base`
    independent = rng.normal(0.0004, 0.01, 500)
    returns = pd.DataFrame({"BASE": base, "DUPLICATE": duplicate, "INDEPENDENT": independent}, index=dates)

    weights = hrp_weights(returns)
    cluster_weight = weights["BASE"] + weights["DUPLICATE"]
    # the redundant pair's combined weight should not dominate the independent asset 2:1 the way
    # naive equal-weighting (66/33) would, since HRP recognizes they're correlated
    assert cluster_weight < 0.66
