"""
Pydantic request and response schemas for the prediction endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class TextPredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="Job posting text to analyse for fraud.",
    )


class ExplanationItem(BaseModel):
    word: str = Field(description="Word or token from the posting.")
    weight: float = Field(
        description="LIME weight. Positive means fraud signal, negative means genuine."
    )


class ReasonGroup(BaseModel):
    id: str
    title: str
    words: List[ExplanationItem]
    attribution: float
    contribution_pct: float


class LimeReasons(BaseModel):
    heading: str
    summary: str
    groups: List[ReasonGroup]


class OcclusionItem(BaseModel):
    sentence: str = Field(description="The sentence that was removed.")
    impact_points: float = Field(
        description="Change in fraud score, in percentage points, when this "
                    "sentence is removed. Positive means the sentence was "
                    "pushing the score up."
    )
    score_without_pct: float = Field(
        description="The fraud score when this sentence is removed."
    )
    share_pct: float = Field(
        description="This sentence's share of all measured movement."
    )


class Occlusion(BaseModel):
    available: bool = Field(
        description="False when the posting is too short to analyse."
    )
    reason: Optional[str] = None
    baseline_pct: Optional[float] = Field(
        default=None, description="Fraud score on the complete posting."
    )
    sentences_tested: Optional[int] = None
    raises: List[OcclusionItem] = Field(
        default_factory=list,
        description="Sentences whose removal lowered the score, so they were "
                    "pushing the posting towards fraud."
    )
    lowers: List[OcclusionItem] = Field(
        default_factory=list,
        description="Sentences whose removal raised the score, so they were "
                    "pushing the posting towards genuine."
    )



class ScoreBreakdownStep(BaseModel):
    sentence: str = Field(description="The factor (sentence) added at this step.")
    step_points: float = Field(
        description="How many percentage points this factor adds to the score, "
                    "measured by removing it from what remains at this step."
    )
    score_after_pct: float = Field(
        description="The running score once this factor is included."
    )


class ScoreBreakdown(BaseModel):
    available: bool = Field(
        description="False when the posting is too short to break down."
    )
    reason: Optional[str] = None
    final_pct: Optional[float] = Field(
        default=None, description="The model's fraud score on the full posting."
    )
    floor_pct: Optional[float] = Field(
        default=None,
        description="The score that remains once every measured factor is "
                    "removed: the model's baseline suspicion on the remainder. "
                    "floor_pct + sum(step_points) equals final_pct within rounding."
    )
    steps: List[ScoreBreakdownStep] = Field(
        default_factory=list,
        description="Factors ordered biggest-first, each with its measured step."
    )
    sentences_tested: Optional[int] = None

class SignalFactor(BaseModel):
    id: str
    title: str
    points: float
    words: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class SignalBreakdown(BaseModel):
    available: bool
    reason: Optional[str] = None
    final_pct: Optional[float] = None
    reference_pct: Optional[float] = None
    reconstruct_pct: Optional[float] = None
    truncated: Optional[bool] = None
    flagged: List[SignalFactor] = Field(default_factory=list)
    stylistic: Optional[SignalFactor] = None
    weak_explanation: Optional[bool] = None


class PatternCategory(BaseModel):
    id: str
    title: str
    detected: List[str]
    explanation: str
    model_attribution_pct: Optional[float] = None
    model_words: List[ExplanationItem] = Field(default_factory=list)


class PatternAnalysis(BaseModel):
    warning: List[PatternCategory]
    genuine: List[PatternCategory]
    warning_summary: str
    genuine_summary: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    fraud_probability: float
    risk_tier: str
    explanation: List[ExplanationItem]
    lime_reasons: LimeReasons
    local_fidelity: Optional[float] = None
    occlusion: Occlusion = Field(
        description="Sentence-level contributions measured by removing each "
                    "sentence and re-running the model."
    )
    score_breakdown: ScoreBreakdown = Field(
        description="Cumulative breakdown that climbs from a baseline floor to "
                    "the final score; the per-factor steps sum to the score."
    )
    pattern_analysis: PatternAnalysis
    signal_breakdown: SignalBreakdown
    input_type: str
    ocr_text: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str