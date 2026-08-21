"""
Prediction endpoints for the Fraud Detector API.

POST /predict/text  — analyse a text job posting
POST /predict/image — analyse an image/screenshot of a job posting

Each request produces two complementary explanations:

  1. LIME attributions (model explanation), from app.models.xai. These
     measure what the model actually responded to, grouped by vocabulary
     with each group's share of total attribution.

  2. Rule-based pattern detection, from app.core.patterns. This detects
     documented fraud indicators in the RAW text, knowledge the model does
     not necessarily contain, and reports how much of the model's
     attribution fell on each detected phrase.

The LIME step runs FIRST so its attributions can be passed into the
pattern detector. Pattern detection never influences the prediction.
"""

import json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.core.config import LIME_NUM_FEATURES
from app.core.preprocessing import clean_text
from app.core.patterns import analyse_posting
from app.core.ocr import extract_text_from_bytes
from app.db.database import get_db
from app.db.schema import Submission, Prediction, Explanation
from app.models.xai import explain_prediction, get_risk_tier
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

    Order matters twice over. The model receives CLEANED text, as it
    always has. Pattern detection receives the RAW text, because the
    cleaning step strips digits and punctuation and would destroy
    patterns like "RM500/day" or phone numbers. And the LIME step runs
    before pattern detection so its attributions can be passed in.
    """
    # Step 1: Clean text for the model
    cleaned = clean_text(raw_text)
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Text is empty after cleaning. Please provide a longer, "
                   "more detailed job posting."
        )

    # Step 2: Predict and explain (LIME layer; includes the prediction)
    explanation_result = explain_prediction(cleaned, num_features=LIME_NUM_FEATURES)

    # Step 3: Pattern detection on the raw text, informed by the model's
    # measured attributions so each detected category can report how much
    # the model actually weighted it.
    pattern_result = analyse_posting(
        raw_text,
        lime_explanation=explanation_result["explanation"],
    )

    # Step 4: Risk tier
    risk = get_risk_tier(explanation_result["fraud_probability"])

    # Step 5: Save to database (both explanation layers stored for auditability)
    submission = Submission(
        input_type=input_type,
        raw_input=ocr_raw if ocr_raw else raw_text,
        clean_text=cleaned,
    )
    db.add(submission)
    db.flush()

    pred_record = Prediction(
        submission_id=submission.id,
        label=explanation_result["label"],
        fraud_probability=explanation_result["fraud_probability"],
        confidence=explanation_result["confidence"],
        risk_tier=risk,
    )
    db.add(pred_record)
    db.flush()

    expl_record = Explanation(
        prediction_id=pred_record.id,
        method="lime+patterns",
        top_features_json=json.dumps({
            "explanation": explanation_result["explanation"],
            "lime_reasons": explanation_result["lime_reasons"],
            "local_fidelity": explanation_result["local_fidelity"],
            "pattern_analysis": pattern_result,
        }),
    )
    db.add(expl_record)
    db.commit()

    # Step 6: Build response
    return PredictionResponse(
        label=explanation_result["label"],
        confidence=explanation_result["confidence"],
        fraud_probability=explanation_result["fraud_probability"],
        risk_tier=risk,
        explanation=explanation_result["explanation"],
        lime_reasons=explanation_result["lime_reasons"],
        local_fidelity=explanation_result["local_fidelity"],
        pattern_analysis=pattern_result,
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
    Returns label, confidence, risk tier, the model's grouped LIME
    reasoning, and rule-based pattern detections with the model's
    measured attribution for each.
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