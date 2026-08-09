"""
Column-role detection heuristics.

Given a raw DataFrame (as loaded from CSV/Excel with no assumptions about
meaning), guess which column plays which role in quantitative-finance terms
(Date, Ticker, Close, Returns, Weight, ...). This is a *suggestion* — the
frontend always lets the user override it before any calculation runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Canonical roles the rest of the engine understands.
ROLES = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "returns",
    "weight",
    "position",
    "quantity",
    "pnl",
    "benchmark",
    "risk_free",
    "portfolio_value",
    "factor",
    "sector",
    "ignore",
]

# name -> role, checked as substring against a lowercased, underscore-normalized header
_NAME_PATTERNS: list[tuple[str, list[str]]] = [
    ("date", [r"^date$", r"^dt$", r"^timestamp$", r"^time$", r"trade_date", r"as_of"]),
    ("ticker", [r"^ticker$", r"^symbol$", r"^security$", r"^asset$", r"^name$", r"^instrument$"]),
    ("adj_close", [r"adj.?close", r"adjusted.?close"]),
    ("close", [r"^close$", r"^px_last$", r"^price$", r"closing.?price"]),
    ("open", [r"^open$"]),
    ("high", [r"^high$"]),
    ("low", [r"^low$"]),
    ("volume", [r"^vol(ume)?$"]),
    ("returns", [r"return", r"^ret$", r"_ret$"]),
    ("weight", [r"weight", r"^wt$", r"^wgt$"]),
    ("position", [r"^position$", r"^side$"]),
    ("quantity", [r"quantity", r"^qty$", r"^shares$", r"^units$"]),
    ("pnl", [r"pnl", r"p&l", r"p_l", r"profit"]),
    ("benchmark", [r"benchmark", r"^bmk$", r"^spx$", r"^spy$", r"^index$"]),
    ("risk_free", [r"risk.?free", r"^rf$", r"^rfr$", r"tbill", r"treasury"]),
    ("portfolio_value", [r"portfolio.?value", r"^nav$", r"^equity$", r"account.?value"]),
    ("sector", [r"^sector$", r"^industry$", r"^gics"]),
]


@dataclass
class ColumnGuess:
    column: str
    role: str
    confidence: float  # 0-1
    reason: str


@dataclass
class StructureGuess:
    format_id: str  # "wide_prices" | "ohlcv_long" | "holdings" | "returns_wide" | "portfolio_level" | "unknown"
    label: str
    description: str
    confidence: float


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _looks_like_dates(series: pd.Series, sample: int = 50) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    s = s.sample(min(sample, len(s)), random_state=0) if len(s) > sample else s
    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def _looks_numeric(series: pd.Series) -> float:
    coerced = pd.to_numeric(series, errors="coerce")
    return float(coerced.notna().mean())


def _looks_categorical_ticker(series: pd.Series) -> float:
    """High score if values are short repeated strings (few unique vs many rows)."""
    s = series.dropna().astype(str)
    if s.empty:
        return 0.0
    unique_ratio = s.nunique() / len(s)
    avg_len = s.str.len().mean()
    score = 0.0
    if unique_ratio < 0.5:
        score += 0.5
    if avg_len <= 6:
        score += 0.3
    if s.str.match(r"^[A-Z0-9.\-]+$").mean() > 0.8:
        score += 0.2
    return min(score, 1.0)


def guess_columns(df: pd.DataFrame) -> list[ColumnGuess]:
    guesses: list[ColumnGuess] = []
    used_roles: set[str] = set()

    for col in df.columns:
        norm = _normalize(col)
        best_role, best_conf, best_reason = "ignore", 0.0, "no strong match"

        # name-based match first (cheap, high precision)
        for role, patterns in _NAME_PATTERNS:
            if role in used_roles:
                continue
            for pat in patterns:
                if re.search(pat, norm):
                    conf = 0.9
                    if role in ("date",) and _looks_like_dates(df[col]) < 0.5:
                        conf = 0.4
                    if conf > best_conf:
                        best_role, best_conf, best_reason = role, conf, f"column name matches /{pat}/"
                    break

        # content-based fallback / corroboration
        if best_role == "ignore" or best_conf < 0.85:
            date_score = _looks_like_dates(df[col])
            if date_score > 0.85 and "date" not in used_roles:
                if date_score > best_conf:
                    best_role, best_conf, best_reason = "date", date_score, "values parse as dates"
            else:
                numeric_score = _looks_numeric(df[col])
                ticker_score = _looks_categorical_ticker(df[col])
                if numeric_score > 0.9 and best_role == "ignore":
                    # ambiguous numeric column — leave as "ignore" (likely a price series
                    # in wide format; handled at structure level, not per-column role)
                    best_role, best_conf, best_reason = "ignore", numeric_score * 0.3, "numeric but role unclear"
                elif ticker_score > 0.6 and "ticker" not in used_roles and best_role == "ignore":
                    best_role, best_conf, best_reason = "ticker", ticker_score, "short repeated string values"

        guesses.append(ColumnGuess(column=col, role=best_role, confidence=round(best_conf, 2), reason=best_reason))
        if best_role != "ignore":
            used_roles.add(best_role)

    return guesses


def guess_structure(df: pd.DataFrame, column_guesses: list[ColumnGuess]) -> StructureGuess:
    roles = {g.column: g.role for g in column_guesses}
    role_set = set(roles.values())
    has_date = "date" in role_set
    has_ticker = "ticker" in role_set
    has_ohlc = {"open", "high", "low", "close"}.issubset(role_set) or "adj_close" in role_set
    has_weight_qty = "weight" in role_set or "quantity" in role_set
    has_returns_named = "returns" in role_set
    has_portfolio_value = "portfolio_value" in role_set

    numeric_unlabelled = [
        c for c, r in roles.items() if r == "ignore" and _looks_numeric(df[c]) > 0.9
    ]

    if has_date and has_portfolio_value:
        return StructureGuess(
            "portfolio_level", "Portfolio-level series",
            "Date + portfolio (and optionally benchmark) value series.", 0.85,
        )
    if has_date and has_ticker and has_weight_qty:
        return StructureGuess(
            "holdings", "Portfolio holdings",
            "Date + Ticker + Weight/Quantity(+Price) — position-level holdings data.", 0.85,
        )
    if has_date and has_ticker and has_ohlc:
        return StructureGuess(
            "ohlcv_long", "Long-format OHLCV",
            "Date + Ticker with Open/High/Low/Close(/Volume) stacked by security.", 0.85,
        )
    if has_date and not has_ticker and len(numeric_unlabelled) >= 1 and has_returns_named:
        return StructureGuess(
            "returns_wide", "Wide return series",
            "Date column with one numeric return column per security.", 0.7,
        )
    if has_date and not has_ticker and len(numeric_unlabelled) >= 1:
        return StructureGuess(
            "wide_prices", "Wide price series",
            "Date column with one numeric price column per security (e.g. Date | AAPL | MSFT | ...).", 0.75,
        )
    return StructureGuess("unknown", "Unrecognized structure", "Could not confidently classify the layout.", 0.2)
