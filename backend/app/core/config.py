"""
Central configuration for the Fraud Detector backend.
All path and environment settings live here.
Other modules import from this file rather than hardcoding values.
"""

import os
from pathlib import Path

# ── Project root (two levels up from this file: app/core/ → app/ → backend/) ─
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# ── Model paths ───────────────────────────────────────────────────────────────
BERT_MODEL_PATH = str(
    PROJECT_ROOT / "ml" / "saved_models" / "bert_fraud_detector_multitask_v3"
)
BASELINE_MODEL_PATH = str(
    PROJECT_ROOT / "ml" / "saved_models" / "baseline_random_forest.joblib"
)
TFIDF_VECTORIZER_PATH = str(
    PROJECT_ROOT / "ml" / "saved_models" / "tfidf_vectorizer.joblib"
)

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{BACKEND_DIR / 'fraud_detector.db'}"

# ── API settings ──────────────────────────────────────────────────────────────
API_TITLE = "Fraud Detector API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Explainable AI API for detecting fraudulent online job postings. "
    "Accepts text or image input and returns a fraud prediction with "
    "LIME-based word-level explanations."
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the React dev server to call the API during development
CORS_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",   # fallback
]

# ── XAI settings ─────────────────────────────────────────────────────────────
DECISION_THRESHOLD = 0.5
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 300

# ── Verify critical paths exist at import time ────────────────────────────────
def verify_paths():
    """Check that model files exist. Called during FastAPI startup."""
    missing = []
    for name, path in [
        ("BERT model", BERT_MODEL_PATH),
        ("Baseline model", BASELINE_MODEL_PATH),
        ("TF-IDF vectorizer", TFIDF_VECTORIZER_PATH),
    ]:
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")
    return missing


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BERT_MODEL_PATH:", BERT_MODEL_PATH)
    print("DATABASE_URL:", DATABASE_URL)
    missing = verify_paths()
    if missing:
        print("WARNING — missing paths:")
        for m in missing:
            print(" ", m)
    else:
        print("All paths verified.")
