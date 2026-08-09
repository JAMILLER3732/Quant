"""
In-memory dataset session store.

The engine runs as a single persistent process (Render/Railway/Fly), so a
process-local dict keyed by dataset_id is sufficient for this stage — no
external database required. Sessions expire after TTL to bound memory use.
Note: this means state is lost on redeploy/restart, and won't survive
horizontal scaling to multiple instances; revisit with Redis if/when that
matters.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import pandas as pd

TTL_SECONDS = 60 * 60 * 4  # 4 hours


@dataclass
class DatasetSession:
    id: str
    filename: str
    df: pd.DataFrame
    role_map: dict[str, str]
    sheet_names: list[str]
    used_sheet: str | None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class DatasetStore:
    def __init__(self) -> None:
        self._data: dict[str, DatasetSession] = {}
        self._lock = Lock()

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._data.items() if now - v.last_used > TTL_SECONDS]
        for k in expired:
            self._data.pop(k, None)

    def create(self, filename: str, df: pd.DataFrame, role_map: dict[str, str],
               sheet_names: list[str], used_sheet: str | None) -> DatasetSession:
        with self._lock:
            self._evict_expired()
            sid = str(uuid.uuid4())
            session = DatasetSession(id=sid, filename=filename, df=df, role_map=role_map,
                                      sheet_names=sheet_names, used_sheet=used_sheet)
            self._data[sid] = session
            return session

    def get(self, dataset_id: str) -> DatasetSession:
        with self._lock:
            session = self._data.get(dataset_id)
            if session is None:
                raise KeyError(f"Dataset '{dataset_id}' not found or expired. Please re-upload.")
            session.last_used = time.time()
            return session

    def update_mapping(self, dataset_id: str, role_map: dict[str, str]) -> DatasetSession:
        session = self.get(dataset_id)
        session.role_map = role_map
        return session


STORE = DatasetStore()
