"""
XAI (Explainable AI) module for the Fraud Detector backend.

All explanations in this module are derived from LIME attributions
measured on the model itself. Nothing is asserted about a posting that
the model did not actually respond to.

How the explanation is produced:

  1. LIME perturbs the input 300 times, removing different words each
     time, and observes how the model's output changes. The resulting
     weights measure each word's real contribution to this prediction.

  2. Weak and grammatical features are filtered out for readability.

  3. The surviving features are grouped into vocabulary categories, and
     each category's share of the total measured attribution is computed
     and reported as a percentage. A category appears only if LIME
     actually weighted a word belonging to it, and its percentage is a
     measurement, not an assigned value.

The category keyword lists are a fixed vocabulary used only to give
groups readable names. They carry no weights and cannot cause a category
to appear on their own: the model's attributions decide what is shown and
how much it counts.

SHAP analysis was conducted separately in ml/notebooks/04_evaluation.ipynb
for offline/batch global feature importance analysis.
"""

import os
import re
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
MAX_WORDS_PER_SIDE = 6
MIN_WORDS_PER_SIDE = 3

# When the model is at least this confident, the opposing side's LIME
# weights are statistical noise, so they are suppressed rather than shown
# as misleading signals.
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

# ── Vocabulary groupings ──────────────────────────────────────────────────────
# These lists only give readable NAMES to groups of words. They contain no
# weights and no scores. A category appears in the explanation only when
# LIME has attributed real weight to one of its words, and the percentage
# shown is computed from those attributions.
#
# Titles describe the KIND of vocabulary the model responded to. They do
# not claim the posting is doing something, because an individual word
# cannot establish intent.

FRAUD_VOCABULARY_GROUPS = [
    {
        "id": "income",
        "title": "Money and earnings vocabulary",
        "keywords": {
            "income", "earn", "earning", "earnings", "salary", "salaries",
            "money", "cash", "dollar", "dollars", "rm", "usd", "inr",
            "myr", "sgd", "commission", "commissions", "commision",
            "bonus", "bonuses", "pay", "paid", "payment", "payments",
            "payout", "profit", "profits", "reward", "rewards", "daily",
            "weekly", "monthly", "fee", "fees", "deposit", "wage",
            "wages", "registration",
        },
    },
    {
        "id": "urgency",
        "title": "Urgency and time-pressure vocabulary",
        "keywords": {
            "urgent", "urgently", "immediately", "immediate", "now",
            "today", "tonight", "hurry", "fast", "quick", "quickly",
            "instant", "instantly", "limited", "asap", "rush", "deadline",
            "expires", "expire", "slots", "spots",
        },
    },
    {
        "id": "financial_details",
        "title": "Financial and identity vocabulary",
        "keywords": {
            "bank", "banking", "account", "accounts", "password",
            "passport", "identity", "nric", "card", "credit", "transfer",
            "upfront", "wallet", "paypal", "details", "copy", "document",
            "documents", "ic",
        },
    },
    {
        "id": "easy_work",
        "title": "Remote and low-effort work vocabulary",
        "keywords": {
            "home", "remote", "easy", "simple", "anywhere", "flexible",
            "freedom", "effortless", "typing", "entry", "part", "online",
        },
    },
    {
        "id": "recruitment",
        "title": "Recruitment and application vocabulary",
        "keywords": {
            "hiring", "hire", "recruit", "recruiting", "recruitment",
            "apply", "join", "interested", "vacancy", "vacancies",
            "positions", "position", "candidates", "wanted", "opening",
            "openings", "register",
        },
    },
    {
        "id": "contact",
        "title": "Contact and messaging vocabulary",
        "keywords": {
            "whatsapp", "telegram", "wechat", "sms", "text", "call",
            "phone", "contact", "reach", "message", "messages", "number",
            "kindly", "send",
        },
    },
]

GENUINE_VOCABULARY_GROUPS = [
    {
        "id": "qualifications",
        "title": "Qualifications and skills vocabulary",
        "keywords": {
            "degree", "bachelor", "bachelors", "masters", "diploma",
            "qualification", "qualifications", "experience", "experienced",
            "skills", "skilled", "proficiency", "proficient", "familiarity",
            "familiar", "knowledge", "certified", "technical", "python",
            "java", "javascript", "sql", "git", "engineering", "engineer",
            "developer", "development", "programming",
        },
    },
    {
        "id": "role",
        "title": "Role and responsibility vocabulary",
        "keywords": {
            "responsibilities", "responsibility", "duties", "role", "roles",
            "maintaining", "maintain", "developing", "develop", "managing",
            "manage", "supporting", "support", "collaborating", "collaborate",
            "features", "projects", "project", "tasks", "team", "teams",
            "reviews", "review", "senior", "junior", "version", "planning",
            "reports", "campaign",
        },
    },
    {
        "id": "employment_terms",
        "title": "Employment terms and benefits vocabulary",
        "keywords": {
            "benefits", "insurance", "annual", "leave", "permanent",
            "contract", "fulltime", "allowance", "allowances", "epf",
            "socso", "medical", "dental", "pension", "competitive",
            "training", "structured", "coverage",
        },
    },
    {
        "id": "process",
        "title": "Hiring process vocabulary",
        "keywords": {
            "interview", "interviews", "assessment", "resume", "cv",
            "application", "applications", "process", "shortlisted",
            "screening", "rounds", "portal", "careers", "submit",
        },
    },
    {
        "id": "company",
        "title": "Company and organisational vocabulary",
        "keywords": {
            "company", "established", "firm", "office", "department",
            "industry", "organisation", "organization", "technologies",
            "solutions", "innovations", "corporate", "professional",
        },
    },
]

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


def _normalise_token(word):
    """
    Lowercase a LIME token and, if it is a currency amount such as
    "RM500" or "USD200", reduce it to its currency prefix so it groups
    with other money vocabulary. Returns the lowercased token otherwise.
    """
    w = word.lower()
    m = re.match(r"^(rm|usd|inr|myr|sgd)\d[\d,]*$", w)
    if m:
        return m.group(1)
    return w


def _group_by_vocabulary(words, groups, generic_title):
    """
    Group LIME features by vocabulary category and compute each group's
    share of the total measured attribution.

    The contribution percentage for each group is:

        sum(|weight| of that group's words) / sum(|weight| of all words)

    This is a measurement of the model's behaviour on this specific
    posting. Words matching no category are placed in an explicit generic
    group rather than forced into one.
    """
    if not words:
        return []

    total = sum(abs(w["weight"]) for w in words)
    if total == 0:
        return []

    buckets = {g["id"]: [] for g in groups}
    generic = []

    for item in words:
        w = _normalise_token(item["word"])
        placed = False
        for g in groups:
            if w in g["keywords"]:
                buckets[g["id"]].append(item)
                placed = True
                break
        if not placed:
            generic.append(item)

    result = []
    for g in groups:
        bucket = buckets[g["id"]]
        if bucket:
            weight_sum = sum(abs(w["weight"]) for w in bucket)
            result.append({
                "id": g["id"],
                "title": g["title"],
                "words": bucket,
                "attribution": round(weight_sum, 4),
                "contribution_pct": round(weight_sum / total * 100, 1),
            })
    if generic:
        weight_sum = sum(abs(w["weight"]) for w in generic)
        result.append({
            "id": "other",
            "title": generic_title,
            "words": generic,
            "attribution": round(weight_sum, 4),
            "contribution_pct": round(weight_sum / total * 100, 1),
        })

    # Order by measured contribution, strongest first.
    result.sort(key=lambda r: r["contribution_pct"], reverse=True)
    return result


def _build_reasons(label, driving_words):
    """
    Build the grouped explanation of the model's decision from the
    filtered LIME features. Every group and every percentage is derived
    from the attributions; no group can appear without measured weight
    behind it.

    The summary names the strongest RECOGNISED category rather than the
    generic bucket, since "other vocabulary" tells the reader nothing.
    When only generic vocabulary is present, that is stated plainly.
    """
    if label == "fraudulent":
        groups = _group_by_vocabulary(
            driving_words,
            FRAUD_VOCABULARY_GROUPS,
            "Other vocabulary the model associated with fraudulent postings",
        )
        heading = "Why the AI flagged this posting"
        direction = "fraud"
    else:
        groups = _group_by_vocabulary(
            driving_words,
            GENUINE_VOCABULARY_GROUPS,
            "Other vocabulary the model associated with genuine postings",
        )
        heading = "Why the AI judged this genuine"
        direction = "genuine"

    named = [g for g in groups if g["id"] != "other"]
    generic = next((g for g in groups if g["id"] == "other"), None)

    if not groups:
        summary = (
            "No individual words carried enough measured weight to explain "
            "this decision."
        )
    elif named:
        top = named[0]
        named_total = round(sum(g["contribution_pct"] for g in named), 1)
        if len(named) == 1:
            summary = (
                f"{top['contribution_pct']}% of the model's attribution came "
                f"from {top['title'].lower()}, the only recognised category "
                f"behind this {direction} decision."
            )
        else:
            summary = (
                f"{named_total}% of the model's attribution came from "
                f"{len(named)} recognised categories, led by "
                f"{top['title'].lower()} at {top['contribution_pct']}%."
            )
        if generic and generic["contribution_pct"] >= 40:
            summary += (
                f" The remaining {generic['contribution_pct']}% came from "
                f"words that fit no known category."
            )
    else:
        summary = (
            "The model's attribution came entirely from general vocabulary "
            "rather than any recognised category."
        )

    return {
        "heading": heading,
        "summary": summary,
        "groups": groups,
    }


def explain_prediction(text: str, num_features: int = 15) -> dict:
    """
    Generate a LIME explanation for a single text prediction.

    The prediction (label, confidence, fraud_probability) is computed
    separately by predict_single() and is NOT affected by anything done
    here. This function only shapes how the model's measured reasoning is
    presented.

    Filtering:
      - Stopwords (grammatical connector words) are removed first.
      - Then words are kept only if their weight is at least
        RELATIVE_KEEP_RATIO of the strongest word on their side.
      - Between MIN_WORDS_PER_SIDE and MAX_WORDS_PER_SIDE words are kept
        per side, when available.
      - When model confidence is at or above COUNTER_SUPPRESS_CONFIDENCE,
        the opposing side's words are below the noise floor and are
        suppressed entirely.

    Grouping:
      - The words driving the predicted class are grouped by vocabulary
        and each group's share of total attribution is computed.

    Local fidelity:
      - LIME fits a simple linear surrogate to approximate the model's
        behaviour around this posting. The returned score (R-squared of
        that surrogate) reports how faithfully the word-level explanation
        represents the model's true local behaviour. Values are typically
        low for saturated, very high-confidence predictions, because
        removing any single word barely changes the output.
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

    # The words driving the predicted class are the ones that explain the
    # decision, so those are what get grouped into reasons.
    driving_words = (
        warning_filtered if prediction["label"] == "fraudulent"
        else safe_filtered
    )
    lime_reasons = _build_reasons(prediction["label"], driving_words)

    # LIME's local fidelity, wrapped defensively so a future LIME change
    # degrades to no score rather than crashing the endpoint.
    try:
        local_fidelity = round(float(exp.score), 4)
    except Exception:
        local_fidelity = None

    return {
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "fraud_probability": prediction["fraud_probability"],
        "explanation": explanation,
        "lime_reasons": lime_reasons,
        "local_fidelity": local_fidelity,
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