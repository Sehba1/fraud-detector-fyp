"""
Health check endpoint.
GET /health — returns API status and model load state.
"""

import os
from fastapi import APIRouter
from app.schemas.prediction import HealthResponse
from app.core.config import API_VERSION, BERT_MODEL_PATH

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns API status and whether the BERT model files are present.
    Used to verify the API is running correctly before sending predictions.
    """
    model_loaded = os.path.exists(
        os.path.join(BERT_MODEL_PATH, "model.safetensors")
    )
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        version=API_VERSION,
    )