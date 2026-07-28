"""
Prediction endpoints for the Fraud Detector API.

POST /predict/text  — analyse a text job posting
POST /predict/image — analyse an image/screenshot of a job posting
"""

import json
import tempfile
import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.core.config import LIME_NUM_FEATURES
from app.core.preprocessing import clean_text
from app.core.ocr import extract_text_from_bytes
from app.db.database import get_db
from app.db.schema import Submission, Prediction, Explanation
from app.models.xai import predict_single, explain_prediction, get_risk_tier
from app.schemas.prediction import TextPredictionRequest, PredictionResponse

router = APIRouter()

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def _run_prediction_pipeline(
    raw_text: str,
    input_type: str,
    db: Session,
    ocr_raw: str | None = None,
) -> PredictionResponse:
    """
    Shared pipeline for both text and image endpoints.
    Cleans text → predicts → explains → saves to DB → returns response.
    """
    # Step 1: Clean text
    cleaned = clean_text(raw_text)
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Text is empty after cleaning. Please provide a longer, "
                   "more detailed job posting."
        )

    # Step 2: Predict
    prediction = predict_single(cleaned)

    # Step 3: Explain
    explanation_result = explain_prediction(cleaned, num_features=LIME_NUM_FEATURES)

    # Step 4: Risk tier
    risk = get_risk_tier(prediction["fraud_probability"])

    # Step 5: Save to database
    submission = Submission(
        input_type=input_type,
        raw_input=ocr_raw if ocr_raw else raw_text,
        clean_text=cleaned,
    )
    db.add(submission)
    db.flush()  # assigns submission.id without committing

    pred_record = Prediction(
        submission_id=submission.id,
        label=prediction["label"],
        fraud_probability=prediction["fraud_probability"],
        confidence=prediction["confidence"],
        risk_tier=risk,
    )
    db.add(pred_record)
    db.flush()

    expl_record = Explanation(
        prediction_id=pred_record.id,
        method="lime",
        top_features_json=json.dumps(explanation_result["explanation"]),
    )
    db.add(expl_record)
    db.commit()

    # Step 6: Build response
    return PredictionResponse(
        label=prediction["label"],
        confidence=prediction["confidence"],
        fraud_probability=prediction["fraud_probability"],
        risk_tier=risk,
        explanation=explanation_result["explanation"],
        input_type=input_type,
        ocr_text=ocr_raw,
    )


@router.post("/predict/text", response_model=PredictionResponse)
async def predict_text(
    request: TextPredictionRequest,
    db: Session = Depends(get_db),
):
    """
    Analyse a text job posting for fraud.

    Accepts raw or HTML job posting text.
    Returns label, confidence, risk tier, and LIME word-level explanation.
    """
    return _run_prediction_pipeline(
        raw_text=request.text,
        input_type="text",
        db=db,
    )


@router.post("/predict/image", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Analyse a screenshot or image of a job posting for fraud.

    Accepts JPEG, PNG, WebP, or BMP image uploads.
    Extracts text via OCR, then analyses for fraud.
    Returns label, confidence, risk tier, LIME explanation, and raw OCR text.
    """
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Supported types: {', '.join(SUPPORTED_IMAGE_TYPES)}"
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    suffix = "." + file.content_type.split("/")[1]
    try:
        ocr_raw = extract_text_from_bytes(image_bytes, suffix=suffix)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

    if not ocr_raw or len(ocr_raw.strip()) < 10:
        raise HTTPException(
            status_code=422,
            detail="OCR could not extract meaningful text from this image. "
                   "Please ensure the image is clear and contains readable text."
        )

    return _run_prediction_pipeline(
        raw_text=ocr_raw,
        input_type="image",
        db=db,
        ocr_raw=ocr_raw,
    )