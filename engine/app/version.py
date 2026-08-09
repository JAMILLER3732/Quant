"""Single source of truth for the engine's version, so main.py (FastAPI app
metadata) and api/routes.py (/api/health) can both reference it without a
circular import between them."""
VERSION = "0.3.0"  # Phase 3: 15 methods (returns/stats, technical, risk, simulation, optimization, factor/econometrics)
