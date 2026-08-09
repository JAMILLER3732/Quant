"""
Optional AI-narrative client for the report generator.

Design goal: report generation must NEVER fail because the AI layer is
unavailable, misconfigured, or out of credits — every report is fully usable
as a deterministic, data-only document. The AI client only ever *adds*
narrative prose on top of numbers that were already computed by the quant
engine; it is never the source of any number in a report.

Configuration (env vars):
  REPORT_AI_API_KEY   - required to enable narrative generation
  REPORT_AI_BASE_URL   - optional; point at a proxy (e.g. an Anthropic-
                         compatible gateway) instead of api.anthropic.com
  REPORT_AI_MODEL      - optional; defaults to "claude-opus-5"

Key-type detection: a native Anthropic key (sk-ant-...) authenticates via
the standard x-api-key header (api_key=). Any other key format (e.g. a
third-party Anthropic-compatible gateway's key) is sent via
Authorization: Bearer (auth_token=) instead, since that's the convention
those gateways use. This lets the same code work whether REPORT_AI_API_KEY
is a direct Anthropic key or a compatible gateway's key, with only the env
vars changing.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"


def _build_client():
    import anthropic

    api_key = os.environ.get("REPORT_AI_API_KEY", "").strip()
    if not api_key:
        return None

    base_url = os.environ.get("REPORT_AI_BASE_URL", "").strip() or None
    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url

    if api_key.startswith("sk-ant-"):
        kwargs["api_key"] = api_key
    else:
        # Third-party Anthropic-compatible gateway convention: Bearer auth.
        kwargs["auth_token"] = api_key

    return anthropic.Anthropic(**kwargs)


def is_configured() -> bool:
    return bool(os.environ.get("REPORT_AI_API_KEY", "").strip())


def model_name() -> str:
    return os.environ.get("REPORT_AI_MODEL", "").strip() or DEFAULT_MODEL


def generate_narrative(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str | None:
    """
    Returns the generated text, or None if the AI layer is unavailable for
    any reason (not configured, auth failure, no credits, network error,
    refusal). Callers must treat None as "fall back to templated text" —
    never as an error to surface to the user as a report-generation failure.
    """
    client = _build_client()
    if client is None:
        return None

    try:
        response = client.messages.create(
            model=model_name(),
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:  # noqa: BLE001 — any AI-layer failure degrades gracefully
        logger.warning("AI narrative generation failed, falling back to templated text: %s", exc)
        return None

    if response.stop_reason == "refusal":
        logger.warning("AI narrative generation refused by safety classifiers.")
        return None

    text_parts = [block.text for block in response.content if block.type == "text"]
    result = "".join(text_parts).strip()
    return result or None
