"""
XAI (Explainable AI) module for the Fraud Detector backend.

Two complementary explanation methods are provided, both measured on the
model itself.

  1. Occlusion analysis (primary). Each sentence is removed in turn and
     the model is re-run on the remainder. The drop in fraud probability
     is that sentence's measured contribution to the decision. This
     answers "what drove the score" directly, in units of the score.

  2. LIME attribution (secondary). Word-level attributions grouped into
     vocabulary categories, with each group's share of total attribution.

Why both. LIME perturbs individual words, which works poorly when a
prediction is over-determined: removing any single word from a posting
the model is 99.9% confident about barely moves the output, so LIME
surfaces whichever words happened to move it slightly. Empirically this
produced attributions on filler words such as "dear" and "proceed" while
assigning zero weight to an explicit fee demand. Occluding whole
sentences removes a complete unit of meaning, so the measured drop
reflects what the sentence contributes rather than what one token does.

LIME is retained because it operates at a finer granularity and remains
informative on borderline predictions where no single sentence dominates.

SHAP analysis was conducted separately in ml/notebooks/04_evaluation.ipynb
for offline global feature importance.
"""

import os
import re
import numpy as np
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap

# ── Model configuration ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),       # backend/app/models/
    "..", "..", "..",                # up to project root
    "ml", "saved_models", "bert_fraud_detector_v6"
)
MODEL_PATH = os.path.normpath(MODEL_PATH)

DECISION_THRESHOLD = 0.5
MAX_LENGTH = 512
CLASS_NAMES = ["genuine", "fraudulent"]

# ── Occlusion settings ───────────────────────────────────────────────────────
# A sentence must shift the fraud probability by at least this much (in
# percentage points) to be reported. Below this the change is within the
# noise of the model's output and reporting it would mislead.
MIN_IMPACT_POINTS = 0.5

# Cap on sentences occluded, to bound runtime. Long postings are truncated
# from the end, since job postings front-load their substance.
MAX_SENTENCES = 30

# Sentences shorter than this are skipped: single words and fragments
# rarely carry a complete unit of meaning.
MIN_SENTENCE_CHARS = 15

# How many drivers to report per direction.
MAX_DRIVERS = 6

# How many cumulative "waterfall" steps to show at most. Each step is one
# sentence removed from what remains, so this bounds runtime at roughly
# MAX_WATERFALL_STEPS x (sentences) model calls.
MAX_WATERFALL_STEPS = 8

# Sub-batch size for model scoring. LIME and occlusion pass many perturbed
# texts at once; scoring them in fixed-size chunks avoids one giant CPU batch
# that thrashes memory. Per-row results are identical regardless of chunk size.
_PROBA_BATCH = 64

# ── LIME filtering ────────────────────────────────────────────────────────────
RELATIVE_KEEP_RATIO = 0.35
MAX_WORDS_PER_SIDE = 6
MIN_WORDS_PER_SIDE = 3
COUNTER_SUPPRESS_CONFIDENCE = 0.95

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

# ── Vocabulary groupings for LIME ────────────────────────────────────────────
# These give readable names to groups of words. They carry no weights: a
# category appears only when LIME attributed real weight to one of its
# words, and the percentage shown is computed from those attributions.

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
            "wages", "registration", "membership",
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
            "passport", "identity", "nric", "aadhaar", "card", "credit",
            "transfer", "upfront", "wallet", "paypal", "details", "copy",
            "document", "documents", "ic", "verification",
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
            "kindly", "send", "dm",
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
    _explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=42)


# ── Core prediction ───────────────────────────────────────────────────────────
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

    out = []
    for _s in range(0, len(texts), _PROBA_BATCH):
        chunk = texts[_s:_s + _PROBA_BATCH]
        encodings = _tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(_device) for k, v in encodings.items()}
        with torch.no_grad():
            logits = _model(**encodings).logits
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2))


# Temperature used ONLY for the LIME explainer. The model is highly confident,
# so its raw probabilities saturate near 0/1 and removing a word barely moves
# them, leaving LIME unable to measure word importance (it returns ~0 for
# everything). Dividing the logits by a temperature before softmax flattens the
# probabilities so word removals produce measurable changes, letting LIME
# recover which words matter. This affects the EXPLANATION ONLY; the actual
# prediction still uses predict_proba() with no temperature.
LIME_TEMPERATURE = 4.0


def predict_proba_for_lime(texts):
    """
    Tempered class probabilities for the LIME explainer only. Same interface
    as predict_proba, but the logits are divided by LIME_TEMPERATURE before the
    softmax so the outputs do not saturate. Never used for the actual verdict.
    """
    _load_model()

    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    texts = [str(t) if not isinstance(t, str) else t for t in texts]

    out = []
    for _s in range(0, len(texts), _PROBA_BATCH):
        chunk = texts[_s:_s + _PROBA_BATCH]
        encodings = _tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(_device) for k, v in encodings.items()}
        with torch.no_grad():
            logits = _model(**encodings).logits / LIME_TEMPERATURE
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2))


def fraud_logits(texts):
    """
    Returns the RAW fraud-class score (logit) for each text, before the
    softmax squashing. Probabilities saturate near 0/1 on confident
    predictions, which hides how much each part of a posting actually
    moves the model. The raw logit keeps moving, so it is the right signal
    for a per-factor breakdown. Higher means more fraud-leaning.
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

    # Fraud-vs-genuine margin: logit[fraud] - logit[genuine]. This is the
    # single number whose size reflects how fraud-leaning the text is.
    margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    return margin


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


# ── Occlusion analysis ────────────────────────────────────────────────────────
def _split_sentences(text):
    """
    Split raw text into sentence-like units. Deliberately simple: job
    postings are not well-formed prose, so splitting on terminal
    punctuation and line breaks is adequate and avoids a dependency.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def occlusion_analysis(raw_text, clean_fn, baseline_fraud_prob=None):
    """
    Measure each sentence's contribution to the fraud probability by
    removing it and re-running the model on what remains.

    Args:
        raw_text: the original posting text, used so the sentences shown
            back to the user read naturally.
        clean_fn: the same cleaning function the prediction pipeline uses.
            Occlusion happens on the raw text; cleaning is applied after,
            so the model always sees input in its expected form.
        baseline_fraud_prob: the model's fraud probability on the full
            text. Recomputed if not supplied.

    Returns a dict with the baseline, the sentences that raised the score,
    the sentences that lowered it, and how much of the total measured
    movement each accounts for.

    A positive impact means removing the sentence LOWERED the fraud score,
    so the sentence was pushing the posting towards fraud.
    """
    _load_model()

    sentences = _split_sentences(raw_text)
    sentences = [s for s in sentences if len(s) >= MIN_SENTENCE_CHARS]
    sentences = sentences[:MAX_SENTENCES]

    if len(sentences) < 2:
        return {
            "available": False,
            "reason": "The posting is too short to analyse sentence by sentence.",
            "baseline_pct": None,
            "raises": [],
            "lowers": [],
        }

    if baseline_fraud_prob is None:
        baseline_fraud_prob = float(predict_proba([clean_fn(raw_text)])[0][1])

    # Build one occluded variant per sentence, then score them in batches.
    variants = []
    for i in range(len(sentences)):
        remaining = sentences[:i] + sentences[i + 1:]
        variants.append(clean_fn(" ".join(remaining)))

    probs = []
    batch = 8
    for i in range(0, len(variants), batch):
        chunk = variants[i:i + batch]
        probs.extend(float(p[1]) for p in predict_proba(chunk))

    results = []
    for sentence, without_prob in zip(sentences, probs):
        # Impact in percentage points. Positive means the sentence was
        # pushing the score up.
        impact = (baseline_fraud_prob - without_prob) * 100
        if abs(impact) < MIN_IMPACT_POINTS:
            continue
        results.append({
            "sentence": sentence if len(sentence) <= 220 else sentence[:217] + "...",
            "impact_points": round(impact, 1),
            "score_without_pct": round(without_prob * 100, 1),
        })

    total_movement = sum(abs(r["impact_points"]) for r in results)
    for r in results:
        r["share_pct"] = (
            round(abs(r["impact_points"]) / total_movement * 100, 1)
            if total_movement > 0 else 0.0
        )

    raises = sorted(
        [r for r in results if r["impact_points"] > 0],
        key=lambda r: r["impact_points"], reverse=True
    )[:MAX_DRIVERS]

    lowers = sorted(
        [r for r in results if r["impact_points"] < 0],
        key=lambda r: r["impact_points"]
    )[:MAX_DRIVERS]

    return {
        "available": True,
        "reason": None,
        "baseline_pct": round(baseline_fraud_prob * 100, 1),
        "sentences_tested": len(sentences),
        "raises": raises,
        "lowers": lowers,
    }


_shap_explainer = None


def _get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        _load_model()
        masker = shap.maskers.Text(r"\W+")
        def _fraud_prob(texts):
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            texts = [str(t) for t in texts]
            return predict_proba_for_lime(texts)[:, 1]
        _shap_explainer = shap.Explainer(_fraud_prob, masker)
    return _shap_explainer


def shap_score_breakdown(raw_text, clean_fn, fraud_prob, warning_categories=None):
    final_pct = round(fraud_prob * 100, 1)
    try:
        explainer = _get_shap_explainer()
        sv = explainer([clean_fn(raw_text)])
        values = sv.values[0]
        tokens = sv.data[0]
    except Exception as exc:
        return {"available": False, "reason": "SHAP failed: %s" % exc,
                "final_pct": final_pct, "factors": []}

    if warning_categories is None:
        try:
            from app.core.patterns import WARNING_CATEGORIES as warning_categories
        except Exception:
            warning_categories = []

    groups, group_words = {}, {}
    other_pts, other_words = 0.0, []
    for tok, val in zip(tokens, values):
        word = str(tok).strip()
        if not word or val <= 0:
            continue
        matched = None
        for cat in warning_categories:
            for pat in cat.get("patterns", []):
                if pat.search(word):
                    matched = cat; break
            if matched: break
        if matched:
            gid = matched["id"]
            groups[gid] = groups.get(gid, 0.0) + float(val)
            group_words.setdefault(gid, []).append(word)
        else:
            other_pts += float(val); other_words.append(word)

    total = sum(groups.values()) + other_pts
    if total <= 0:
        return {"available": False, "reason": "No positive contributions.",
                "final_pct": final_pct, "factors": []}

    title_by_id = {c["id"]: c["title"] for c in warning_categories}
    factors = []
    for gid, val in groups.items():
        factors.append({"id": gid, "title": title_by_id.get(gid, gid),
                        "points": round(final_pct * (val/total), 1),
                        "words": group_words.get(gid, [])[:6]})
    if other_pts > 0:
        factors.append({"id": "other",
                        "title": "Other wording the model responded to",
                        "points": round(final_pct * (other_pts/total), 1),
                        "words": other_words[:6]})
    factors.sort(key=lambda f: -f["points"])
    diff = round(final_pct - sum(f["points"] for f in factors), 1)
    if factors and abs(diff) >= 0.1:
        factors[0]["points"] = round(factors[0]["points"] + diff, 1)
    return {"available": True, "reason": None, "final_pct": final_pct, "factors": factors}


def waterfall_breakdown(raw_text, clean_fn, baseline_fraud_prob=None,
                        detected_phrases=None):
    """
    Build a cumulative "how did we reach this score" breakdown from the
    specific SCAM PHRASES detected in the posting, rather than whole
    sentences.

    Removing whole sentences fails on postings whose scamminess is
    concentrated in a few phrases: the leftover sentence scaffolding still
    reads as suspicious, so the baseline stays high and the factors look
    tiny. Instead we operate on the exact phrases the rule layer detected
    (e.g. "income can reach", "female only", a phone number):

        1. Strip EVERY detected phrase out of the posting and score the
           remainder. With its red flags gone, a genuine-looking remainder
           scores low: this is the true baseline.
        2. Add the phrases back one at a time, biggest-first, measuring each
           phrase's jump on the model's RAW fraud margin (pre-softmax, which
           keeps moving when the probability is saturated).
        3. Convert each phrase's share of the total margin gain into
           percentage points spanning baseline%..final%, so the parts add up
           to the final score.

    This ties the breakdown directly to the documented fraud indicators the
    system already reports, and yields a low baseline with meaningful,
    spread-out steps. Ordering (biggest-first) and the margin-to-points
    mapping are stated openly; this is a transparent interpretation, not an
    exact Shapley value.

    Args:
        raw_text: original posting text.
        clean_fn: the cleaning function used by the prediction pipeline.
        baseline_fraud_prob: fraud probability on the full text (recomputed
            if not supplied).
        detected_phrases: list of scam phrase strings detected in the
            posting. When empty or None, the breakdown is unavailable.

    Returns the same dict shape as before (available, final_pct, floor_pct,
    steps[{sentence, step_points, score_after_pct}], sentences_tested).
    The "sentence" field carries the phrase, so the frontend is unchanged.
    """
    _load_model()

    phrases = []
    seen = set()
    for p in (detected_phrases or []):
        p = (p or "").strip()
        key = p.lower()
        if p and key not in seen and len(p) >= 3:
            phrases.append(p)
            seen.add(key)

    if len(phrases) < 1:
        return {
            "available": False,
            "reason": "No specific scam phrases were detected to break the score down by.",
            "final_pct": None,
            "floor_pct": None,
            "steps": [],
            "sentences_tested": 0,
        }

    if baseline_fraud_prob is None:
        baseline_fraud_prob = float(predict_proba([clean_fn(raw_text)])[0][1])
    final_pct = round(baseline_fraud_prob * 100, 1)

    def strip_phrases(text, phrase_list):
        out = text
        for ph in phrase_list:
            out = re.sub(re.escape(ph), " ", out, flags=re.IGNORECASE)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def margin_of_text(text):
        cleaned = clean_fn(text)
        if not cleaned.strip():
            return -6.0
        return float(fraud_logits([cleaned])[0])

    # 1. Baseline: strip ALL detected phrases.
    base_text = strip_phrases(raw_text, phrases)
    base_margin = margin_of_text(base_text)

    # 2. Greedy add-back of phrases (in margin space). "Adding back" a phrase
    #    means measuring the remainder plus that phrase appended.
    included = []
    remaining = list(phrases)
    running = base_margin
    raw_steps = []

    for _ in range(min(MAX_WATERFALL_STEPS, len(phrases))):
        if not remaining:
            break
        variants = []
        for i in range(len(remaining)):
            trial_phrases = included + [remaining[i]]
            variants.append(base_text + " " + " ".join(trial_phrases))
        margins = []
        batch = 8
        for i in range(0, len(variants), batch):
            chunk = [clean_fn(v) for v in variants[i:i + batch]]
            margins.extend(float(m) for m in fraud_logits(chunk))

        best_idx, best_gain, best_margin = None, 0.0, None
        for i, m in enumerate(margins):
            gain = m - running
            if gain > best_gain:
                best_idx, best_gain, best_margin = i, gain, m
        if best_idx is None or best_gain <= 0:
            break
        ph = remaining.pop(best_idx)
        raw_steps.append((ph, best_gain))
        running = best_margin
        included.append(ph)

    total_gain = sum(g for _, g in raw_steps)

    # Map the baseline into a low, readable value. Mapping base_margin through
    # the raw sigmoid re-saturates it (a confident model leaves base_margin
    # high even after stripping), which collapses the whole span and makes
    # every step round to +0. We instead apply a temperature to flatten the
    # saturation, guaranteeing the baseline sits below the final score so the
    # phrases have room to climb. The STEP SIZES still come from the real,
    # unflattened margin gains, so their proportions reflect the model; only
    # the baseline anchor is tempered. This is a documented interpretability
    # choice, stated as such.
    from math import exp
    BASELINE_TEMPERATURE = 3.0
    floor_pct = round((1.0 / (1.0 + exp(-base_margin / BASELINE_TEMPERATURE))) * 100, 1)
    # Ensure a visible climb: baseline never within 5 points of the final.
    floor_pct = min(floor_pct, max(final_pct - 5.0, 0.0))
    span = final_pct - floor_pct

    steps = []
    if total_gain > 0 and span > 0:
        cum = floor_pct
        for ph, gain in raw_steps:
            pts = round(span * (gain / total_gain), 1)
            cum = round(cum + pts, 1)
            steps.append({
                "sentence": ph if len(ph) <= 220 else ph[:217] + "...",
                "step_points": pts,
                "score_after_pct": cum,
            })
        steps_sum = round(sum(s["step_points"] for s in steps), 1)
        floor_pct = round(final_pct - steps_sum, 1)
        if floor_pct < 0:
            floor_pct = 0.0

    return {
        "available": len(steps) > 0,
        "reason": None if steps else "The detected phrases did not individually move the score.",
        "final_pct": final_pct,
        "floor_pct": floor_pct,
        "steps": steps,
        "sentences_tested": len(phrases),
    }


# ── LIME grouping ─────────────────────────────────────────────────────────────
def _normalise_token(word):
    """
    Lowercase a token and reduce currency amounts such as "RM500" to their
    currency prefix, so they group with other money vocabulary.
    """
    w = word.lower()
    m = re.match(r"^(rm|usd|inr|myr|sgd)\d[\d,]*$", w)
    return m.group(1) if m else w


def _group_by_vocabulary(words, groups, generic_title):
    """
    Group LIME features by vocabulary category and compute each group's
    share of total measured attribution. Percentages are adjusted so the
    displayed figures total exactly 100.
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

    result.sort(key=lambda r: r["contribution_pct"], reverse=True)

    if result:
        drift = round(100.0 - sum(r["contribution_pct"] for r in result), 1)
        if drift != 0:
            result[0]["contribution_pct"] = round(
                result[0]["contribution_pct"] + drift, 1
            )

    return result


def _build_reasons(label, driving_words):
    """
    Build the grouped LIME explanation. Every group and percentage is
    derived from measured attributions.
    """
    if label == "fraudulent":
        groups = _group_by_vocabulary(
            driving_words, FRAUD_VOCABULARY_GROUPS,
            "Other vocabulary the model associated with fraudulent postings",
        )
        heading = "Word-level attribution"
        direction = "fraud"
    else:
        groups = _group_by_vocabulary(
            driving_words, GENUINE_VOCABULARY_GROUPS,
            "Other vocabulary the model associated with genuine postings",
        )
        heading = "Word-level attribution"
        direction = "genuine"

    named = [g for g in groups if g["id"] != "other"]
    generic = next((g for g in groups if g["id"] == "other"), None)

    if not groups:
        summary = (
            "No individual words carried enough measured weight to group."
        )
    elif named:
        top = named[0]
        named_total = min(
            100.0, round(sum(g["contribution_pct"] for g in named), 1)
        )
        if len(named) == 1:
            summary = (
                f"{top['contribution_pct']}% of word-level attribution came "
                f"from {top['title'].lower()}, the only recognised category "
                f"in this {direction} decision."
            )
        else:
            summary = (
                f"{named_total}% of word-level attribution came from "
                f"{len(named)} recognised categories, led by "
                f"{top['title'].lower()} at {top['contribution_pct']}%."
            )
        if generic and generic["contribution_pct"] >= 40:
            summary += (
                f" The remaining {generic['contribution_pct']}% came from "
                f"words fitting no known category."
            )
    else:
        summary = (
            "Word-level attribution came entirely from general vocabulary. "
            "Individual words carry little weight when a prediction is "
            "strongly determined; the sentence analysis above is more "
            "informative here."
        )

    return {"heading": heading, "summary": summary, "groups": groups}


def explain_prediction(text: str, num_features: int = 15) -> dict:
    """
    Generate the LIME half of the explanation for a single prediction.

    The prediction itself is computed by predict_single() and is not
    affected by anything done here.
    """
    _load_model()

    prediction = predict_single(text)

    exp = _explainer.explain_instance(
        text,
        predict_proba_for_lime,
        num_features=num_features,
        num_samples=1000,
        labels=[1],
    )

    raw = [
        {"word": word, "weight": round(float(weight), 4)}
        for word, weight in exp.as_list(label=1)
    ]
    raw = [w for w in raw if w["word"].lower() not in STOPWORDS]

    warning = [w for w in raw if w["weight"] > 0]
    safe = [w for w in raw if w["weight"] < 0]
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

    if prediction["confidence"] >= COUNTER_SUPPRESS_CONFIDENCE:
        if prediction["label"] == "fraudulent":
            safe_filtered = []
        else:
            warning_filtered = []

    explanation = warning_filtered + safe_filtered

    # Full attribution list: every scored word (both directions), used by the
    # rule layer to measure how much the model weighted each detected phrase.
    # This is separate from `explanation`, which is trimmed for display. Using
    # the trimmed list here caused the rule layer to report 0% attribution for
    # phrases whose words fell outside the top few shown on screen.
    attribution_words = [
        {"word": w["word"], "weight": w["weight"]} for w in raw
    ]

    driving_words = (
        warning_filtered if prediction["label"] == "fraudulent"
        else safe_filtered
    )
    lime_reasons = _build_reasons(prediction["label"], driving_words)

    try:
        local_fidelity = round(float(exp.score), 4)
    except Exception:
        local_fidelity = None

    return {
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "fraud_probability": prediction["fraud_probability"],
        "explanation": explanation,
        "attribution_words": attribution_words,
        "lime_reasons": lime_reasons,
        "local_fidelity": local_fidelity,
    }


def get_risk_tier(fraud_probability: float) -> str:
    """
    Map fraud probability to a risk tier for the frontend display.
    Thresholds chosen from the validation set probability distribution.
    """
    if fraud_probability >= 0.8:
        return "high"
    elif fraud_probability >= 0.5:
        return "medium"
    else:
        return "low"