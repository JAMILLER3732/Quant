from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MappingUpdateRequest(BaseModel):
    role_map: dict[str, str]


class CalculateRequest(BaseModel):
    role_map: dict[str, str] | None = None  # optional override; falls back to session's stored mapping
    params: dict[str, Any] = {}
