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


class ReasonGroup(BaseModel):
    id: str = Field(description="Stable identifier for this vocabulary group.")
    title: str = Field(description="Human-readable group title.")
    words: List[ExplanationItem] = Field(
        description="LIME features from this prediction that belong to this group."
    )
    attribution: float = Field(
        description="Sum of absolute LIME weights for this group's words."
    )
    contribution_pct: float = Field(
        description="This group's share of the model's total measured attribution, "
                    "as a percentage."
    )


class LimeReasons(BaseModel):
    heading: str = Field(description="Section heading for the model's reasoning.")
    summary: str = Field(
        description="One-sentence summary of the model's measured attribution."
    )
    groups: List[ReasonGroup] = Field(
        description="LIME features grouped by vocabulary, ordered by measured "
                    "contribution."
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
    model_attribution_pct: Optional[float] = Field(
        default=None,
        description="Share of the model's total LIME attribution that fell on the "
                    "words in this category's matched snippets. Zero means the "
                    "pattern is present in the text but the model did not respond "
                    "to it."
    )
    model_words: List[ExplanationItem] = Field(
        default_factory=list,
        description="The LIME features underlying this category's attribution figure."
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
    lime_reasons: LimeReasons = Field(
        description="LIME attributions grouped by vocabulary, with each group's "
                    "measured share of the model's decision."
    )
    local_fidelity: Optional[float] = Field(
        default=None,
        description="R-squared of the LIME surrogate model, indicating how faithfully "
                    "the word-level explanation approximates the model's local behaviour."
    )
    pattern_analysis: PatternAnalysis = Field(
        description="Rule-based detection of documented fraud indicators, each "
                    "reporting the model's measured attribution. Detection never "
                    "influences the prediction."
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