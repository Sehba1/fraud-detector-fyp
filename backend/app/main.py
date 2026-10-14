"""
FastAPI application entry point for the Fraud Detector API.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

From inside the backend/ directory with venv activated.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    CORS_ORIGINS,
    verify_paths,
)
from app.db.database import init_db
from app.api.predict import router as predict_router
from app.api.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    Runs once when the server starts, cleans up when it stops.
    """
    # Startup
    print("Starting Fraud Detector API...")

    # Verify model files exist before accepting requests
    missing = verify_paths()
    if missing:
        print("WARNING: The following model files are missing:")
        for path in missing:
            print(f"  {path}")
    else:
        print("All model paths verified.")

    # Initialize database tables
       # Initialize database tables
    init_db()
    print("Database ready.")

    # Warm up the model and XAI once, at startup, so the first /predict request
    # does not pay the cold BERT build inline. Nothing here changes any XAI
    # value; it only moves the existing one-time loading from request time to
    # boot time.
    print("Warming up V3 model and XAI backgrounds...")
    try:
        import app.models.multitask_infer as mi
        from app.models.faithful_xai import _load_backgrounds
        mi._load()                      # build MultiTaskBert + load V3 weights once (cached)
        _load_backgrounds()             # cache the 10 genuine backgrounds once
        mi.multitask_fraud_prob("warm up", clean_fn=None)  # JIT the forward path
        print("Warm-up complete. Model and backgrounds are cached.")
    except Exception as e:
        print(f"WARNING: warm-up failed ({e}); model will load lazily on first request.")

    print("API running — visit http://localhost:8000/docs for interactive docs.")

    yield  # server runs here

    # Shutdown
    print("Shutting down Fraud Detector API.")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

# CORS — allow the React frontend (Vite dev server) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router, tags=["Health"])
app.include_router(predict_router, tags=["Predictions"])