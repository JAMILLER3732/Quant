"""File ingestion: turn raw uploaded bytes into a pandas DataFrame."""
from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd


class IngestionError(Exception):
    """Raised when a file cannot be parsed at all (wrong format, corrupt, empty)."""


@dataclass
class IngestResult:
    df: pd.DataFrame
    sheet_names: list[str]
    used_sheet: str | None
    file_kind: str  # "csv" | "excel"


def sniff_kind(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xls", ".xlsm")):
        return "excel"
    if lower.endswith(".csv"):
        return "csv"
    raise IngestionError(
        f"Unsupported file extension for '{filename}'. Please upload a .csv, .xlsx, or .xls file."
    )


def ingest_file(filename: str, content: bytes, sheet_name: str | None = None) -> IngestResult:
    if not content:
        raise IngestionError("The uploaded file is empty.")

    kind = sniff_kind(filename)
    buf = io.BytesIO(content)

    if kind == "csv":
        try:
            df = pd.read_csv(buf, sep=None, engine="python")
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Could not parse CSV file: {exc}") from exc
        if df.empty or df.shape[1] == 0:
            raise IngestionError("The CSV file was parsed but contains no columns/rows.")
        return IngestResult(df=df, sheet_names=[], used_sheet=None, file_kind="csv")

    # excel
    try:
        xls = pd.ExcelFile(buf)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(
            f"Could not open Excel file — it may be corrupt or password protected: {exc}"
        ) from exc

    sheets = xls.sheet_names
    if not sheets:
        raise IngestionError("The Excel file has no sheets.")

    target_sheet = sheet_name if sheet_name in sheets else sheets[0]
    try:
        df = xls.parse(target_sheet)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Could not read sheet '{target_sheet}': {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise IngestionError(f"Sheet '{target_sheet}' was parsed but contains no columns/rows.")

    # Excel exports frequently leave trailing "phantom used-range" columns/rows
    # that aren't truly empty — they contain only whitespace or a stray
    # non-breaking space (\xa0) in a handful of cells, a formatting artifact
    # rather than real data (observed in a real Bloomberg-exported workbook).
    # Blank those out before the emptiness check, or they survive as a
    # near-100%-null junk column that downstream heuristics can misclassify.
    df = df.apply(lambda col: col.map(
        lambda v: pd.NA if isinstance(v, str) and v.strip(" \xa0\t\n\r") == "" else v
    ))
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    return IngestResult(df=df, sheet_names=sheets, used_sheet=target_sheet, file_kind="excel")
