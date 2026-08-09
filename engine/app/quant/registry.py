"""
Central registry of all QuantMethod implementations.

Adding a new method = write a QuantMethod subclass in app/quant/methods/
and add one line here. The API and frontend discover everything else
(metadata, params, requirements) generically through the base contract.
"""
from __future__ import annotations

from app.quant.base import QuantMethod
from app.quant.methods.correlation_analysis import CorrelationAnalysisMethod
from app.quant.methods.efficient_frontier import EfficientFrontierMethod
from app.quant.methods.ewma_crossover import EwmaCrossoverMethod
from app.quant.methods.mean_reversion_backtest import MeanReversionBacktestMethod
from app.quant.methods.monte_carlo_gbm import MonteCarloGbmMethod
from app.quant.methods.performance_dashboard import PerformanceDashboardMethod
from app.quant.methods.returns_descriptive import ReturnsDescriptiveMethod
from app.quant.methods.rolling_zscore import RollingZScoreMethod
from app.quant.methods.stress_testing import StressTestingMethod
from app.quant.methods.var_cvar import VarCvarMethod

_METHOD_CLASSES: list[type[QuantMethod]] = [
    # Phase 1
    ReturnsDescriptiveMethod,
    EwmaCrossoverMethod,
    RollingZScoreMethod,
    PerformanceDashboardMethod,
    # Phase 2
    MonteCarloGbmMethod,
    EfficientFrontierMethod,
    CorrelationAnalysisMethod,
    VarCvarMethod,
    StressTestingMethod,
    MeanReversionBacktestMethod,
]

REGISTRY: dict[str, QuantMethod] = {cls.id: cls() for cls in _METHOD_CLASSES}


def get_method(method_id: str) -> QuantMethod:
    if method_id not in REGISTRY:
        raise KeyError(f"Unknown method id '{method_id}'.")
    return REGISTRY[method_id]


def list_methods() -> list[dict]:
    return [m.metadata() for m in REGISTRY.values()]
