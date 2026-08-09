"""
Monte Carlo simulation primitives. Every simulator here is fully vectorized
(NumPy array ops over [n_simulations, n_steps]) — no per-path Python loops.
"""
from __future__ import annotations

import numpy as np

from app.quant.calc import TRADING_DAYS_PER_YEAR


def gbm_paths(
    s0: float, mu: float, sigma: float, horizon_days: int, n_sims: int,
    steps_per_day: int = 1, seed: int | None = None,
) -> np.ndarray:
    """
    Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW.
    Exact (log-Euler) solution per step avoids discretization bias:
        S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z), Z ~ N(0,1).

    Returns an array of shape (n_sims, n_steps+1) including the initial price
    at column 0, fully vectorized (no per-path loop).
    """
    n_steps = horizon_days * steps_per_day
    dt = 1.0 / (TRADING_DAYS_PER_YEAR * steps_per_day)
    rng = np.random.default_rng(seed)

    z = rng.standard_normal((n_sims, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = s0 * np.exp(log_paths)

    full = np.empty((n_sims, n_steps + 1))
    full[:, 0] = s0
    full[:, 1:] = paths
    return full


def percentile_bands(paths: np.ndarray, percentiles: tuple[float, ...] = (5, 25, 50, 75, 95)) -> dict[float, np.ndarray]:
    """paths: (n_sims, n_steps+1). Returns {percentile: array of shape (n_steps+1,)}."""
    return {p: np.percentile(paths, p, axis=0) for p in percentiles}
