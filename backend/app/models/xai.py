"""
XAI (Explainable AI) module for the Fraud Detector backend.
Provides LIME-based explanations for individual predictions.

LIME is used for real-time explanations (fast enough for live API responses).
SHAP analysis was conducted separately in ml/notebooks/04_evaluation.ipynb
for offline/batch global feature importance analysis.
"""

import os
import numpy as np
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer

# ── Model configuration ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),       # backend/app/models/
    "..", "..", "..",                # up to project root
    "ml", "saved_models", "bert_fraud_detector"
)
MODEL_PATH = os.path.normpath(MODEL_PATH)

DECISION_THRESHOLD = 0.5
MAX_LENGTH = 512
CLASS_NAMES = ["genuine", "fraudulent"]

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
# Model and tokenizer are loaded once on first use, not at import time.
# This avoids a ~2-3 second startup cost when the module is imported
# during testing or for non-prediction endpoints.
_tokenizer = None
_model = None
_device = None
_explainer = None


def _load_model():
    """Load model and tokenizer if not already loaded."""
    global _tokenizer, _model, _device, _explainer

    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

    _model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.to(_device)
    _model.eval()

    _explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        random_state=42,
    )


# ── Core prediction function ──────────────────────────────────────────────────
def predict_proba(texts):
    """
    Returns class probabilities for a list of texts.
    Shape: (n_texts, 2) — [genuine_prob, fraud_prob] per row.
    Compatible with both LIME and SHAP.
    """
    _load_model()

    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()

    texts = [str(t) if not isinstance(t, str) else t for t in texts]

    encodings = _tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encodings = {k: v.to(_device) for k, v in encodings.items()}

    with torch.no_grad():
        logits = _model(**encodings).logits

    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


# ── Public API ────────────────────────────────────────────────────────────────
def predict_single(text: str) -> dict:
    """
    Predict fraud probability for a single text.

    Returns:
        {
            "label": "fraudulent" | "genuine",
            "confidence": float,          # probability of the predicted class
            "fraud_probability": float,   # raw fraud class probability
        }
    """
    _load_model()
    probs = predict_proba([text])[0]
    fraud_prob = float(probs[1])
    label = "fraudulent" if fraud_prob >= DECISION_THRESHOLD else "genuine"
    confidence = fraud_prob if label == "fraudulent" else float(probs[0])

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "fraud_probability": round(fraud_prob, 4),
    }


def explain_prediction(text: str, num_features: int = 10) -> dict:
    """
    Generate a LIME explanation for a single text prediction.

    Returns:
        {
            "label": str,
            "fraud_probability": float,
            "explanation": [{"word": str, "weight": float}, ...]
        }
    """
    _load_model()

    prediction = predict_single(text)

    exp = _explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,
        num_samples=300,       # reduced from notebook's 500 for API latency
        labels=[1],
    )

    explanation = [
        {"word": word, "weight": round(float(weight), 4)}
        for word, weight in exp.as_list(label=1)
    ]

    return {
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "fraud_probability": prediction["fraud_probability"],
        "explanation": explanation,
    }


def get_risk_tier(fraud_probability: float) -> str:
    """
    Map fraud probability to a risk tier for the frontend display.

    Thresholds chosen based on validation set probability distribution.
    """
    if fraud_probability >= 0.8:
        return "high"
    elif fraud_probability >= 0.5:
        return "medium"
    else:
        return "low"