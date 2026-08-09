"""
Turns a ReportData structure (already-computed, real numbers) into narrative
prose — either AI-written (grounded strictly in the provided numbers) or a
plain templated fallback when the AI layer isn't configured/available.

The AI system prompt is explicit that every number it writes about must come
from the provided data block, and that it must not introduce new statistics,
price targets, or recommendations that weren't computed by the engine.
"""
from __future__ import annotations

import json

from app.reports.ai_client import generate_narrative, is_configured
from app.reports.builder import ReportData

SYSTEM_PROMPT = """You are a quantitative research analyst writing the narrative sections of an \
institutional-style research note. You will be given ONLY the real, already-computed statistics for a \
security or portfolio, produced by a Python quantitative engine.

Strict rules:
1. Use ONLY the numbers provided below. Never invent, estimate, or infer a statistic that isn't given to you.
2. Never give a price target, a buy/sell/hold recommendation, or any forward-looking prediction presented as fact.
3. Explicitly distinguish description (what the data shows) from any forward-looking statement (always caveat it \
as not a guarantee).
4. Be precise and quantitative in your prose — reference the actual figures given.
5. Write in a professional, measured tone appropriate for an institutional research note. No hype, no absolutes.
6. Output plain prose organized under the section headers requested. No markdown headers with '#', use plain \
text section titles as instructed.
"""


def _stats_block(data: ReportData) -> str:
    lines = [f"Report scope: {data.scope} — {data.target}", f"Observations: {data.n_observations}"]
    if data.date_range:
        lines.append(f"Date range: {data.date_range[0]} to {data.date_range[1]}")
    for section in data.sections:
        lines.append(f"\n[{section.method_name}]")
        lines.append(json.dumps(section.stats, indent=2, default=str))
    return "\n".join(lines)


def generate_report_narrative(data: ReportData) -> dict[str, str]:
    """Returns {"executive_summary": ..., "analysis": ..., "limitations": ...}.
    Falls back to templated (non-AI) text for any section the AI layer can't produce."""
    if not is_configured():
        return _templated_narrative(data)

    stats_block = _stats_block(data)
    user_prompt = (
        f"Here is the computed data for this report:\n\n{stats_block}\n\n"
        "Write three sections, each 2-4 sentences, plain prose, no markdown formatting:\n\n"
        "EXECUTIVE SUMMARY: A concise overview of what the data shows.\n\n"
        "ANALYSIS: A more detailed discussion connecting the different statistics above "
        "(e.g. how volatility relates to the Sharpe ratio, what the drawdown implies, "
        "how correlation affects diversification if applicable).\n\n"
        "LIMITATIONS: What this analysis does NOT tell the reader, grounded in the specific "
        "methodology limitations of the methods used.\n\n"
        "Separate the three sections with a line containing only '---'."
    )

    text = generate_narrative(SYSTEM_PROMPT, user_prompt, max_tokens=1200)
    if text is None:
        return _templated_narrative(data)

    parts = [p.strip() for p in text.split("---")]
    fallback = _templated_narrative(data)
    return {
        "executive_summary": parts[0] if len(parts) > 0 and parts[0] else fallback["executive_summary"],
        "analysis": parts[1] if len(parts) > 1 and parts[1] else fallback["analysis"],
        "limitations": parts[2] if len(parts) > 2 and parts[2] else fallback["limitations"],
        "ai_generated": "true",
    }


def _templated_narrative(data: ReportData) -> dict[str, str]:
    summary_bits = []
    for section in data.sections:
        headline_keys = list(section.stats.items())[:3]
        formatted = ", ".join(f"{k}: {v}" for k, v in headline_keys)
        summary_bits.append(f"{section.method_name} — {formatted}")

    executive_summary = (
        f"This report covers {data.target} over {data.n_observations} observations"
        + (f" from {data.date_range[0]} to {data.date_range[1]}" if data.date_range else "")
        + f". It combines {len(data.sections)} quantitative analyses: "
        + "; ".join(s.method_name for s in data.sections) + "."
    )
    analysis = "Key figures by method: " + " | ".join(summary_bits) + \
        " See each section below for the full statistics, methodology, and chart."
    limitations = (
        "This report is generated entirely from the uploaded historical data using documented, "
        "testable quantitative methods — it describes what the data shows historically and does not "
        "constitute a forecast, price target, or investment recommendation. See each section's "
        "'Limitations' notes for method-specific caveats."
    )
    return {"executive_summary": executive_summary, "analysis": analysis, "limitations": limitations,
            "ai_generated": "false"}
