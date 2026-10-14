"""
Prediction endpoints for the Fraud Detector API.

POST /predict/text  — analyse a text job posting
POST /predict/image — analyse an image/screenshot of a job posting

Each request produces three complementary explanations:

  1. Occlusion analysis. Each sentence is removed in turn and the model
     re-run on the remainder. The drop in fraud probability is that
     sentence's measured contribution, expressed in the same units as
     the score itself. This is the primary explanation.

  2. LIME attribution. Word-level shares of total attribution, shown as
     supporting detail.

  3. Rule-based pattern detection on the raw text, reporting documented
     fraud indicators alongside the model's measured attribution to them.

Occlusion runs one model pass per sentence, so it adds time proportional
to posting length. MAX_SENTENCES in app/models/xai.py bounds this.
"""
from app.models.faithful_xai import explain as faithful_explain
import json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.core.preprocessing import clean_text
from app.core.patterns import analyse_posting
from app.core.ocr import extract_text_from_bytes
from app.db.database import get_db
from app.db.schema import Submission, Prediction, Explanation
from app.models.xai import get_risk_tier
from app.models.multitask_infer import multitask_fraud_prob
from app.models.faithful_xai import explain as faithful_explain
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
    Shared pipeline for both endpoints.

    The model receives cleaned text. Occlusion and pattern detection both
    receive the RAW text, so the sentences and phrases shown back to the
    user read naturally and so digits and punctuation survive.
    """
    cleaned = clean_text(raw_text)
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Text is empty after cleaning. Please provide a longer, "
                   "more detailed job posting."
        )

    # V3 prediction (the verdict). Unchanged.
    mt_prob = multitask_fraud_prob(raw_text, clean_text)
    label = "fraudulent" if mt_prob >= 0.5 else "genuine"
    confidence = mt_prob if mt_prob >= 0.5 else (1.0 - mt_prob)
    risk = get_risk_tier(mt_prob)

    # Rule-based detection, kept fully separate from the model explanation.
    # No LIME is passed, so no model-derived attribution is written into the
    # rule categories (model_attribution_pct stays None).
    pattern_result = analyse_posting(raw_text)

    # The single model-driven explanation: faithful interventional Shapley on
    # the frozen V3 classifier (reference + factor contributions = V3 score).
    signal_breakdown = faithful_explain(raw_text)

    # LIME, occlusion and waterfall are superseded by the faithful breakdown
    # and are no longer computed per request. Their response fields are kept
    # present (empty / not computed) so the frontend contract does not break.
    empty_lime = {
        "heading": "Model explanation",
        "summary": "Superseded by the faithful factor breakdown (see signal_breakdown).",
        "groups": [],
    }
    empty_occlusion = {
        "available": False,
        "reason": "Not computed; superseded by the faithful factor breakdown.",
        "baseline_pct": None,
        "sentences_tested": None,
        "raises": [],
        "lowers": [],
    }
    empty_score_breakdown = {
        "available": False,
        "reason": "Not computed; superseded by the faithful factor breakdown.",
        "final_pct": round(mt_prob * 100, 2),
        "floor_pct": None,
        "steps": [],
        "sentences_tested": None,
    }

    submission = Submission(
        input_type=input_type,
        raw_input=ocr_raw if ocr_raw else raw_text,
        clean_text=cleaned,
    )
    db.add(submission)
    db.flush()

    pred_record = Prediction(
        submission_id=submission.id,
        label=label,
        fraud_probability=mt_prob,
        confidence=confidence,
        risk_tier=risk,
    )
    db.add(pred_record)
    db.flush()

    expl_record = Explanation(
        prediction_id=pred_record.id,
        method="faithful_shapley_v3+patterns",
        top_features_json=json.dumps({
            "signal_breakdown": signal_breakdown,
            "pattern_analysis": pattern_result,
        }),
    )
    db.add(expl_record)
    db.commit()

    return PredictionResponse(
        label=label,
        confidence=confidence,
        fraud_probability=mt_prob,
        risk_tier=risk,
        explanation=[],
        lime_reasons=empty_lime,
        local_fidelity=None,
        occlusion=empty_occlusion,
        score_breakdown=empty_score_breakdown,
        signal_breakdown=signal_breakdown,
        pattern_analysis=pattern_result,
        input_type=input_type,
        ocr_text=ocr_raw,
    )


@router.post("/predict/text", response_model=PredictionResponse)
async def predict_text(
    request: TextPredictionRequest,
    db: Session = Depends(get_db),
):
    """Analyse a text job posting for fraud."""
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
    """Analyse a screenshot or image of a job posting for fraud."""
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