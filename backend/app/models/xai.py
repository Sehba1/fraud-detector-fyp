"""
XAI (Explainable AI) module for the Fraud Detector backend.

This module provides the model-centric half of the system's hybrid
explanation framework: LIME-based word-level attributions that explain
why the BERT model made its prediction.

The knowledge-based half (rule-based scam pattern analysis of the posting
content) lives separately in app/core/patterns.py and is completely
independent of the model.

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

# ── Explanation filtering ─────────────────────────────────────────────────────
# LIME weight magnitudes vary with model confidence, so a fixed absolute
# cutoff fails. We keep words whose weight is at least RELATIVE_KEEP_RATIO
# of the strongest word on their own side.
RELATIVE_KEEP_RATIO = 0.35
MAX_WORDS_PER_SIDE = 5
MIN_WORDS_PER_SIDE = 3

# When the model is at least this confident, the opposing side's LIME
# weights are statistical noise, so they are suppressed rather than shown
# as misleading "signals".
COUNTER_SUPPRESS_CONFIDENCE = 0.95

# Grammatical connector words carry no meaning for a human explanation and
# are removed from the displayed word lists. This mirrors the stopword
# removal already used for the TF-IDF baselines and is standard XAI
# readability practice.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "as",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "have", "has", "had",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "it", "its", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "they", "them", "his", "her",
    "our", "your", "their", "my", "me", "us",
    "not", "no", "nor", "up", "out", "off", "over", "under",
    "again", "further", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "too", "very", "just", "also", "into", "about",
    "s", "t", "re", "ve", "ll", "d", "m",
}

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
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


def explain_prediction(text: str, num_features: int = 15) -> dict:
    """
    Generate a LIME explanation for a single text prediction.

    The prediction (label, confidence, fraud_probability) is computed
    separately by predict_single() and is NOT affected by anything done
    here. This function only shapes how the word-level evidence is
    presented.

    Filtering:
      - Stopwords (grammatical connector words) are removed first.
      - Then words are kept only if their weight is at least
        RELATIVE_KEEP_RATIO of the strongest word on their side.
      - Between MIN_WORDS_PER_SIDE and MAX_WORDS_PER_SIDE words are shown
        per side, when available.
      - When model confidence is at or above COUNTER_SUPPRESS_CONFIDENCE,
        the opposing side's words are below the noise floor and are
        suppressed entirely.
    """
    _load_model()

    prediction = predict_single(text)

    exp = _explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,
        num_samples=300,
        labels=[1],
    )

    # Full, unfiltered list of (word, weight) from LIME
    raw = [
        {"word": word, "weight": round(float(weight), 4)}
        for word, weight in exp.as_list(label=1)
    ]

    # Remove grammatical stopwords from the displayed explanation
    raw = [w for w in raw if w["word"].lower() not in STOPWORDS]

    # Split into warning (positive) and safe (negative) words
    warning = [w for w in raw if w["weight"] > 0]
    safe = [w for w in raw if w["weight"] < 0]

    # Sort each side by strength (largest absolute weight first)
    warning.sort(key=lambda w: w["weight"], reverse=True)
    safe.sort(key=lambda w: abs(w["weight"]), reverse=True)

    def filter_side(words):
        if not words:
            return []
        strongest = abs(words[0]["weight"])
        cutoff = strongest * RELATIVE_KEEP_RATIO
        kept = [w for w in words if abs(w["weight"]) >= cutoff]
        if len(kept) < MIN_WORDS_PER_SIDE:
            kept = words[:MIN_WORDS_PER_SIDE]
        return kept[:MAX_WORDS_PER_SIDE]

    warning_filtered = filter_side(warning)
    safe_filtered = filter_side(safe)

    # If the model is extremely confident, the opposing side's words are
    # statistical noise; drop them so the evidence shown is honest.
    if prediction["confidence"] >= COUNTER_SUPPRESS_CONFIDENCE:
        if prediction["label"] == "fraudulent":
            safe_filtered = []
        else:
            warning_filtered = []

    explanation = warning_filtered + safe_filtered

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