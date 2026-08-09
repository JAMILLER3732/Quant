from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.data.column_detection import guess_columns, guess_structure
from app.data.ingestion import IngestionError, ingest_file
from app.data.reshape import ReshapeError, price_panel
from app.data.validation import run_validation
from app.quant.registry import REGISTRY, get_method, list_methods
from app.schemas import CalculateRequest, MappingUpdateRequest
from app.store import STORE
from app.version import VERSION

router = APIRouter(prefix="/api")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB
PREVIEW_ROWS = 25


def sanitize(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python and replace
    NaN/Inf with None, so the response is plain, valid JSON."""
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):  # numpy scalar (np.float64, np.int64, np.bool_, ...)
        return sanitize(obj.item())
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def json_response(payload: dict[str, Any]) -> JSONResponse:
    """Bypass FastAPI/pydantic's automatic response serialization (which chokes
    on numpy types) — sanitize() already reduces everything to native types."""
    return JSONResponse(content=sanitize(payload))


def _df_preview(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> list[dict[str, Any]]:
    preview = df.head(n).copy()
    for col in preview.columns:
        if preview[col].dtype.kind in "Mm":
            preview[col] = preview[col].astype(str)
    return sanitize(preview.where(pd.notnull(preview), None).to_dict(orient="records"))


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": VERSION, "methods": str(len(REGISTRY))}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), sheet_name: str | None = Form(None)) -> dict[str, Any]:
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)}MB upload limit.")
    try:
        result = ingest_file(file.filename or "upload", content, sheet_name=sheet_name)
    except IngestionError as exc:
        raise HTTPException(400, str(exc)) from exc

    column_guesses = guess_columns(result.df)
    structure = guess_structure(result.df, column_guesses)
    role_map = {g.column: g.role for g in column_guesses if g.role != "ignore"}

    session = STORE.create(
        filename=file.filename or "upload", df=result.df, role_map=role_map,
        sheet_names=result.sheet_names, used_sheet=result.used_sheet,
    )

    quality = run_validation(result.df, role_map)

    return json_response({
        "dataset_id": session.id,
        "filename": session.filename,
        "sheet_names": result.sheet_names,
        "used_sheet": result.used_sheet,
        "n_rows": int(result.df.shape[0]),
        "n_columns": int(result.df.shape[1]),
        "columns": list(result.df.columns),
        "column_guesses": [
            {"column": g.column, "role": g.role, "confidence": g.confidence, "reason": g.reason}
            for g in column_guesses
        ],
        "structure_guess": {
            "format_id": structure.format_id, "label": structure.label,
            "description": structure.description, "confidence": structure.confidence,
        },
        "role_map": role_map,
        "preview_rows": _df_preview(result.df),
        "quality_report": quality.to_dict(),
    })


@router.post("/datasets/{dataset_id}/mapping")
def update_mapping(dataset_id: str, body: MappingUpdateRequest) -> dict[str, Any]:
    try:
        session = STORE.update_mapping(dataset_id, body.role_map)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    quality = run_validation(session.df, session.role_map)
    return json_response({"role_map": session.role_map, "quality_report": quality.to_dict()})


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        session = STORE.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    quality = run_validation(session.df, session.role_map)
    return json_response({
        "dataset_id": session.id, "filename": session.filename,
        "n_rows": int(session.df.shape[0]), "n_columns": int(session.df.shape[1]),
        "columns": list(session.df.columns), "role_map": session.role_map,
        "preview_rows": _df_preview(session.df), "quality_report": quality.to_dict(),
    })


@router.get("/methods")
def methods() -> dict[str, Any]:
    return json_response({"methods": list_methods()})


@router.get("/datasets/{dataset_id}/methods/{method_id}/requirements")
def method_requirements(dataset_id: str, method_id: str) -> dict[str, Any]:
    try:
        session = STORE.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        method = get_method(method_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    check = method.check_requirements(session.role_map, session.df)

    security_options: list[str] = []
    try:
        panel, _ = price_panel(session.df, session.role_map)
        security_options = list(panel.columns)
    except ReshapeError:
        pass

    return json_response({
        "method_id": method_id,
        "satisfied": check.satisfied,
        "missing": check.missing,
        "warnings": check.warnings,
        "dynamic_param_options": {"security": security_options},
    })


@router.post("/datasets/{dataset_id}/calculate/{method_id}")
def calculate(dataset_id: str, method_id: str, body: CalculateRequest) -> dict[str, Any]:
    try:
        session = STORE.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        method = get_method(method_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    role_map = body.role_map or session.role_map
    check = method.check_requirements(role_map, session.df)
    if not check.satisfied:
        raise HTTPException(422, detail={
            "error": "requirements_not_met",
            "message": f"'{method.name}' cannot be calculated from the current data mapping.",
            "missing": check.missing,
        })

    try:
        result = method.calculate(session.df, role_map, body.params)
    except ValueError as exc:
        raise HTTPException(422, detail={"error": "calculation_error", "message": str(exc)}) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail={"error": "internal_error", "message": f"Calculation failed: {exc}"}) from exc

    return json_response({
        "method_id": method_id,
        "figure": result.figure,
        "stats": result.stats,
        "tables": result.tables,
        "csv_rows": result.series_csv_rows,
        "warnings": result.warnings,
        "notes": result.notes,
        "params_used": body.params,
    })
