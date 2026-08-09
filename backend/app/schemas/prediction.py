"""
Pydantic request and response schemas for the prediction endpoints.
FastAPI uses these for automatic validation, serialization, and API docs.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── Request schemas ───────────────────────────────────────────────────────────

class TextPredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="Job posting text to analyse for fraud.",
        examples=["We are hiring urgently. Work from home. Send bank details to apply."],
    )


# ── Sub-schemas used inside responses ────────────────────────────────────────

class ExplanationItem(BaseModel):
    word: str = Field(description="Word or token from the posting.")
    weight: float = Field(
        description="LIME weight. Positive means fraud signal, negative means genuine signal."
    )


class PatternCategory(BaseModel):
    id: str = Field(description="Stable identifier for this pattern category.")
    title: str = Field(description="Human-readable category title.")
    detected: List[str] = Field(
        description="Actual text snippets from the posting that matched this category."
    )
    explanation: str = Field(
        description="Plain-language explanation of why this pattern matters."
    )


class PatternAnalysis(BaseModel):
    warning: List[PatternCategory] = Field(
        description="Scam warning pattern categories found in the posting."
    )
    genuine: List[PatternCategory] = Field(
        description="Legitimate-posting pattern categories found in the posting."
    )
    warning_summary: str = Field(
        description="One-sentence summary of the warning side."
    )
    genuine_summary: str = Field(
        description="One-sentence summary of the genuine side."
    )


# ── Response schemas ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    label: str = Field(description="'fraudulent' or 'genuine'.")
    confidence: float = Field(description="Probability of the predicted class (0-1).")
    fraud_probability: float = Field(description="Raw fraud class probability (0-1).")
    risk_tier: str = Field(description="'low', 'medium', or 'high'.")
    explanation: List[ExplanationItem] = Field(
        description="Filtered word-level LIME evidence (model explanation)."
    )
    pattern_analysis: PatternAnalysis = Field(
        description="Rule-based scam pattern analysis of the posting content. "
                    "Independent of the model and never influences the prediction."
    )
    input_type: str = Field(description="'text' or 'image'.")
    ocr_text: Optional[str] = Field(
        default=None,
        description="Raw OCR-extracted text (only present for image submissions)."
    )


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' if the API is running correctly.")
    model_loaded: bool = Field(description="Whether the BERT model is loaded.")
    version: str = Field(description="API version string.")