from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.version import VERSION

app = FastAPI(
    title="Quant Engine API",
    description="Python quantitative-finance calculation engine: upload -> validate -> calculate -> chart.",
    version=VERSION,
)

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "quant-engine", "status": "running", "docs": "/docs"}
