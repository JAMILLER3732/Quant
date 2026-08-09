"""
The Quant Method framework.

Every quantitative methodology in the platform (EWMA crossover, Sharpe
dashboard, Z-score bands, ...) is a subclass of `QuantMethod` implementing a
uniform contract:

    Metadata -> RequiredInputs -> Params -> Validation -> Calculation
    -> Statistics -> Visualization -> Explanation

This lets the API and frontend enumerate, describe, validate, and run any
method generically, and lets new methods be added without touching routing
or UI code — see app/quant/registry.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go

Role = str  # one of app.data.column_detection.ROLES


@dataclass
class RequiredInput:
    role: Role
    label: str
    min_series: int = 1  # e.g. efficient frontier needs >=2 securities mapped to this role
    required: bool = True
    note: str = ""


@dataclass
class ParamSpec:
    name: str
    label: str
    type: Literal["int", "float", "select", "bool", "string"]
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[dict[str, Any]] | None = None  # [{"value":..., "label":...}]
    description: str = ""


@dataclass
class RequirementCheck:
    satisfied: bool
    missing: list[str] = field(default_factory=list)   # human-readable reasons
    warnings: list[str] = field(default_factory=list)


@dataclass
class MethodResult:
    figure: dict[str, Any]                 # plotly figure as JSON-able dict (data+layout)
    stats: dict[str, Any]                  # headline numbers, e.g. {"Sharpe Ratio": 1.23, ...}
    tables: dict[str, Any] = field(default_factory=dict)  # named tabular results or secondary figures
    series_csv_rows: list[dict[str, Any]] = field(default_factory=list)    # flat rows for CSV/Excel export
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class QuantMethod(ABC):
    id: str
    name: str
    category: str
    difficulty: Literal["beginner", "intermediate", "advanced"] = "intermediate"

    description: str = ""
    what_it_calculates: str = ""
    why_use_it: str = ""
    methodology: str = ""          # markdown, may include LaTeX-ish notation in $...$
    assumptions: list[str] = []
    limitations: list[str] = []

    required_inputs: list[RequiredInput] = []
    params: list[ParamSpec] = []

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "difficulty": self.difficulty,
            "description": self.description,
            "what_it_calculates": self.what_it_calculates,
            "why_use_it": self.why_use_it,
            "methodology": self.methodology,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "required_inputs": [vars(r) for r in self.required_inputs],
            "params": [vars(p) for p in self.params],
        }

    def check_requirements(self, role_map: dict[str, str], df: pd.DataFrame) -> RequirementCheck:
        """Default structural check: every declared role must be present enough times.
        Subclasses may override for cross-field logic (e.g. needs >=2 tickers)."""
        missing: list[str] = []
        role_counts: dict[str, int] = {}
        for col, role in role_map.items():
            role_counts[role] = role_counts.get(role, 0) + 1

        for req in self.required_inputs:
            count = role_counts.get(req.role, 0)
            if req.required and count < req.min_series:
                if req.min_series > 1:
                    missing.append(
                        f"'{req.label}' requires at least {req.min_series} mapped column(s) with role "
                        f"'{req.role}', found {count}. {req.note}".strip()
                    )
                else:
                    missing.append(f"Missing required data: {req.label} (role '{req.role}'). {req.note}".strip())

        return RequirementCheck(satisfied=len(missing) == 0, missing=missing)

    @abstractmethod
    def calculate(self, df: pd.DataFrame, role_map: dict[str, str], params: dict[str, Any]) -> MethodResult:
        ...

    # -- shared helpers -------------------------------------------------
    @staticmethod
    def fig_to_dict(fig: go.Figure) -> dict[str, Any]:
        return fig.to_plotly_json()
