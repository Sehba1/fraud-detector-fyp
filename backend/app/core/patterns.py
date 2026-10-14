"""
Rule-based scam pattern detection for the Fraud Detector backend.

This module detects documented recruitment-fraud indicators in the raw
posting text. It is knowledge-based, not model-based: the phrase lists
encode published scam indicators and consumer guidance, which is
knowledge the trained model does not necessarily contain.

It does NOT score or weight anything. Each detected category reports:

  - the actual text snippets that matched, as evidence
  - the share of the model's LIME attribution that fell on those words

The second figure is measured from the model, not assigned here, so a
category can be detected in the text while the model attributed nothing
to it. That disagreement is displayed rather than hidden: it shows where
the model's learned behaviour diverges from documented fraud indicators.

Important design facts:

  - Detection never influences the prediction, the confidence, or the
    risk tier.
  - It runs on the RAW input text (or raw OCR output), lightly normalised
    to lowercase with collapsed whitespace. It does not use the BERT
    cleaning pipeline, because that pipeline strips digits and punctuation
    and would destroy patterns like "RM500/day" or phone numbers.
  - Negation handling. Two mechanisms coexist:
      1. Fixed-width negative lookbehinds embedded in a handful of
         GENUINE patterns (e.g. (?<!no )\binterviews?\b), preserved for
         backward compatibility.
      2. A sentence-scoped negation filter for warning categories that
         match noun-phrase scam signals a legitimate posting commonly
         disclaims — money_requests ("no registration fee", "we do not
         charge a processing fee") and identity_requests ("we never ask
         applicants for their bank details"). Those categories carry
         `"negation_sensitive": True` and every match in them is dropped
         when a negator token appears in the same sentence, within
         `_NEG_WINDOW` tokens immediately preceding the match.
    Categories that MATCH ON NEGATION BY DESIGN — notably
    low_requirements ("no experience", "no interview") — are NOT marked
    negation-sensitive and are unaffected.
  - The pattern list is a maintained knowledge base. It is expected to
    grow as new scam variants are encountered; advance-fee "membership
    fee" offer-letter scams were added after one was observed in testing.
"""

import re


def _c(pattern: str):
    return re.compile(pattern)


# ── Sentence-scoped negation filter ──────────────────────────────────────────
# Some warning categories match noun-phrase scam signals — a demand for a fee,
# a deposit, bank details, an ID document — that legitimate postings routinely
# disclaim ("no registration fee", "we do not ask for bank details", "we never
# require your passport"). Without negation handling those disclaimers falsely
# fire as warnings. Other categories match on negation BY DESIGN
# (low_requirements: "no experience", "no interview") and MUST NOT be filtered.
# The filter is therefore opt-in per category via `negation_sensitive: True`.
#
# For opted-in matches we look inside the sentence containing the match at the
# `_NEG_WINDOW` tokens immediately preceding it, and drop the match when any
# of those tokens is a negator. This is surface negation, not full NLP: it
# introduces no new dependency and its scope is small and predictable.

_SENT_DELIMS = ".!?\n"
_NEG_WINDOW = 6
_SIMPLE_NEGATORS = frozenset({
    "no", "not", "never", "without", "cannot", "none", "neither", "nor",
})
_CONTRACTION_NEG = re.compile(r"n['’]t$")
_TOKENIZE = re.compile(r"[a-z][a-z'’]*")


def _prev_sentence_start(text: str, pos: int) -> int:
    """First-char index of the sentence containing `pos`."""
    best = -1
    for ch in _SENT_DELIMS:
        i = text.rfind(ch, 0, pos)
        if i > best:
            best = i
    return best + 1 if best >= 0 else 0


def _is_negated(text: str, match_start: int) -> bool:
    """
    True when a negator occurs within `_NEG_WINDOW` tokens immediately
    preceding `match_start` in the same sentence.

    Handles:
      no / not / never / without / cannot / none / neither / nor
      contractions ending in n't (don't, doesn't, won't, didn't, isn't, ...)

    Sentence scope means a negator in one sentence does NOT suppress a
    match in a different sentence: "No registration fee. Pay a joining fee."
    keeps `joining fee` detected.
    """
    sent_start = _prev_sentence_start(text, match_start)
    prefix = text[sent_start:match_start]
    tokens = _TOKENIZE.findall(prefix.lower())
    for tok in tokens[-_NEG_WINDOW:]:
        if tok in _SIMPLE_NEGATORS:
            return True
        if _CONTRACTION_NEG.search(tok):
            return True
    return False


# ── Money-context anchor for ambiguous asset terms ───────────────────────────
# `bitcoin` / `usdt` / `crypto(currency)` are legitimate domain terminology in
# fintech and blockchain postings ("cryptocurrency exchange", "crypto
# ecosystem") and are NOT, on their own, a request for the applicant's money.
# They are only treated as a money-request warning when the SAME sentence also
# contains a payment-action word. Generic; no company- or posting-specific
# exceptions, and every other pattern is untouched.
_MONEY_CONTEXT = re.compile(
    r"\b(?:pay|paid|payment|payments|payable|send|deposit|deposits|transfer|"
    r"transfers|wire|fee|fees|charge|charges|purchase|buy|invest|investment|"
    r"investing|wallet|remit)\b"
)


def _sentence_span(text: str, pos: int) -> str:
    """The sentence containing `pos`, used to require a money-context anchor in
    the same sentence as an ambiguous asset term."""
    start = _prev_sentence_start(text, pos)
    end = len(text)
    for ch in _SENT_DELIMS:
        i = text.find(ch, pos)
        if i != -1 and i < end:
            end = i
    return text[start:end]


# ── Warning pattern categories ────────────────────────────────────────────────
WARNING_CATEGORIES = [
    {
        "id": "money_requests",
        "title": "Requests for money or financial details",
        # Negation-sensitive: legitimate postings routinely explicitly
        # disclaim these ("no registration fee", "we do not charge a
        # processing fee", "we never ask for bank details").
        "negation_sensitive": True,
        "explanation": (
            "This posting asks for a fee, payment, or banking details. "
            "Legitimate employers do not ask applicants to pay to secure "
            "a job, internship, or offer."
        ),
        "patterns": [
            _c(r"\bregistration fee\b"),
            _c(r"\bprocessing fee\b"),
            _c(r"\btraining fee\b"),
            _c(r"\bmembership fees?\b"),
            _c(r"\bjoining fee\b"),
            _c(r"\bonboarding fee\b"),
            _c(r"\badmission fee\b"),
            _c(r"\bone.?time fee\b"),
            _c(r"\bfee of (?:rs\.?|rm|usd|inr|\$)?\s?\d[\d,]*"),
            _c(r"\bfees? to confirm\b"),
            _c(r"\bclick here to pay\b"),
            _c(r"\bdeposit\b"),
            _c(r"\bbank (?:details?|account|information)\b"),
            _c(r"\bupfront (?:payment|fee)\b"),
            _c(r"\btop.?up\b"),
            _c(r"\binvestment (?:opportunity|plan|scheme|return)\b"),
        ],
        # Ambiguous crypto/asset terms: counted only when the same sentence
        # also contains a payment-action word (see _MONEY_CONTEXT). Legitimate
        # blockchain/fintech postings that merely mention cryptocurrency as
        # domain terminology no longer trigger a money-request warning.
        "context_anchored_patterns": [
            _c(r"\bbitcoin\b"),
            _c(r"\busdt\b"),
            _c(r"\bcrypto(?:currency)?\b"),
        ],
    },
    {
        "id": "identity_requests",
        "title": "Personal identity information requests",
        # Negation-sensitive: "we do not require your passport", "we never
        # ask applicants for personal details".
        "negation_sensitive": True,
        "explanation": (
            "This posting asks for identity documents or personal "
            "information early in the process, which is a common tactic in "
            "identity theft and recruitment scams."
        ),
        "patterns": [
            _c(r"\bpassport\b"),
            _c(r"\bnric\b"),
            _c(r"\bic (?:number|copy)\b"),
            _c(r"\bidentity card\b"),
            _c(r"\bpersonal (?:details?|information)\b"),
        ],
    },
    {
        "id": "unrealistic_income",
        "title": "Unrealistic income promises",
        "explanation": (
            "This posting promises unusually high or fast income with "
            "little information about the work involved, a pattern "
            "frequently used to lure victims."
        ),
        "patterns": [
            _c(r"(?:rm|usd|\$)\s?\d[\d,]*\s?(?:/|per\s?|a\s?)day"),
            _c(r"\bdaily (?:pay|payment|income|salary)\b"),
            _c(r"\bweekly (?:pay|payment|payout)\b"),
            _c(r"\beasy money\b"),
            _c(r"\bfast cash\b"),
            _c(r"\bquick money\b"),
            _c(r"\bincome can reach\b"),
            _c(r"\bunlimited income\b"),
            _c(r"\bguaranteed income\b"),
            _c(r"\bpassive income\b"),
        ],
    },
    {
        "id": "external_contact",
        "title": "Contact through unofficial channels",
        "explanation": (
            "This posting asks applicants to communicate through personal "
            "messaging apps or direct phone contact instead of official "
            "company recruitment channels, which is commonly observed in "
            "recruitment scams."
        ),
        "patterns": [
            _c(r"\btelegram\b"),
            _c(r"\bwhats\s?app\b"),
            _c(r"\bwechat\b"),
            _c(r"\bviber\b"),
            _c(r"\bdirect message\b"),
            _c(r"\bdm (?:us|me)\b"),
            _c(r"\btext (?:us|me) (?:at|on)\b"),
            _c(r"(?:resume|cv) to\s*:?\s*\+?[\d\-\s]{7,}"),
        ],
    },
    {
        "id": "low_requirements",
        "title": "Suspiciously low entry requirements",
        # NOT negation-sensitive. These patterns MATCH ON NEGATION BY
        # DESIGN: "no experience", "no interview", "no qualifications" ARE
        # the scam signal. Applying negation filtering here would erase
        # the whole category.
        "explanation": (
            "This posting advertises extremely low entry requirements, "
            "which frequently appear in fraudulent recruitment "
            "advertisements."
        ),
        "patterns": [
            _c(r"\bno experience\b"),
            _c(r"\bno interview\b"),
            _c(r"\bno qualifications?\b"),
            _c(r"\bno skills? (?:needed|required)\b"),
            _c(r"\banyone can (?:apply|join|do)\b"),
            _c(r"\bstudents? welcome\b"),
            _c(r"\bno resume\b"),
        ],
    },
    {
        "id": "pressure",
        "title": "Pressure and urgency tactics",
        "explanation": (
            "This posting uses urgency language that pushes applicants to "
            "act quickly without proper verification."
        ),
        "patterns": [
            _c(r"\burgent(?:ly)?\b"),
            _c(r"\bimmediate(?:ly)? (?:hiring|joining|start)\b"),
            _c(r"\bapply (?:now|today|tonight)\b"),
            _c(r"\blimited (?:positions?|slots?|spots?|seats?|vacancies)\b"),
            _c(r"\bhurry\b"),
            _c(r"\bact fast\b"),
            _c(r"\bfirst come,? first serve(?:d)?\b"),
            _c(r"\btoday only\b"),
        ],
    },
    {
        "id": "targeting",
        "title": "Unprofessional or discriminatory targeting",
        "explanation": (
            "This posting targets applicants by gender or age in a way "
            "that professional employers avoid, and that is frequently "
            "seen in fraudulent local job advertisements."
        ),
        "patterns": [
            _c(r"\b(?:female|male|ladies|girls|boys)\s?only\b"),
            _c(r"\bage (?:below|under) \d+\b"),
        ],
    },
]

# ── Genuine pattern categories ────────────────────────────────────────────────
GENUINE_CATEGORIES = [
    {
        "id": "benefits",
        "title": "Standard employee benefits",
        "explanation": (
            "The posting lists realistic employee benefits typically "
            "offered by legitimate companies."
        ),
        "patterns": [
            _c(r"\bmedical\b"),
            _c(r"\binsurance\b"),
            _c(r"\bannual leave\b"),
            _c(r"\bepf\b"),
            _c(r"\bsocso\b"),
            _c(r"\bdental\b"),
            _c(r"\bperformance bonus\b"),
            _c(r"\ballowances?\b"),
            _c(r"\bpension\b"),
        ],
    },
    {
        "id": "formal_process",
        "title": "Formal hiring process",
        "explanation": (
            "The posting describes a structured application and interview "
            "process, which is characteristic of genuine employers."
        ),
        "patterns": [
            _c(r"(?<!no )\binterviews?\b"),
            _c(r"\bshortlisted\b"),
            _c(r"\bresume\b"),
            _c(r"\bcv\b"),
            _c(r"\bassessment\b"),
            _c(r"\bscreening\b"),
            _c(r"(?<!not )\bapply through\b"),
        ],
    },
    {
        "id": "qualifications",
        "title": "Professional qualifications required",
        "explanation": (
            "The posting specifies concrete qualifications, skills, or "
            "experience, which scam postings rarely include."
        ),
        "patterns": [
            _c(r"\bbachelor'?s?\b"),
            _c(r"\bdegree\b"),
            _c(r"\bdiploma\b"),
            _c(r"\bmasters?\b"),
            _c(r"\byears? of (?:professional |relevant |working )?experience\b"),
            _c(r"(?<!no )\bexperience (?:required|in|with)\b"),
            _c(r"\bproficien(?:t|cy)\b"),
            _c(r"\bpython\b"),
            _c(r"\bjava(?:script)?\b"),
            _c(r"\bsql\b"),
        ],
    },
    {
        "id": "company_profile",
        "title": "Professional company presence",
        "explanation": (
            "The posting includes company background information, which is "
            "typical of legitimate job advertisements."
        ),
        "patterns": [
            _c(r"\babout us\b"),
            _c(r"\bcompany (?:overview|profile|background)\b"),
            _c(r"\bour (?:company|team|mission)\b"),
            _c(r"\bestablished\b"),
            _c(r"\bsdn\.? bhd\.?\b"),
            _c(r"\bcareers? (?:page|portal|site|website)\b"),
        ],
    },
    {
        "id": "clear_role",
        "title": "Clear role and responsibilities",
        "explanation": (
            "The employer clearly explains the role, duties, and "
            "expectations, which fraudulent postings usually leave vague."
        ),
        "patterns": [
            _c(r"\bresponsibilit(?:y|ies)\b"),
            _c(r"(?<!no )(?<!fee )\brequirements?\b"),
            _c(r"\bjob (?:description|scope)\b"),
            _c(r"\bduties\b"),
            _c(r"\byou will (?:design|develop|manage|maintain|prepare|handle|support|work)\b"),
        ],
    },
]


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace. Keeps digits and punctuation."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _attribution_for_snippets(snippets, lime_index, lime_total):
    """
    Measure how much of the model's LIME attribution fell on the words
    inside these matched snippets.

    Returns (percentage_of_total_attribution, list_of_matched_lime_words).

    A category can match text while the model attributed nothing to it.
    In that case this returns 0.0, which is a real finding and is
    displayed rather than hidden.
    """
    if not lime_index or lime_total == 0:
        return None, []

    seen = {}
    for snippet in snippets:
        for token in re.findall(r"[a-z]+", snippet.lower()):
            if token in lime_index and token not in seen:
                seen[token] = lime_index[token]

    if not seen:
        return 0.0, []

    weight_sum = sum(abs(v["weight"]) for v in seen.values())
    words = [
        {"word": v["word"], "weight": v["weight"]}
        for v in seen.values()
    ]
    words.sort(key=lambda w: abs(w["weight"]), reverse=True)
    return round(weight_sum / lime_total * 100, 1), words


def _scan(text, categories, lime_index, lime_total):
    """
    Run every category's patterns against the text. A category is included
    only if at least one pattern matched, and it carries the actual matched
    snippets as evidence plus the model's measured attribution for those
    words.

    Categories flagged `negation_sensitive: True` have every candidate
    match tested with `_is_negated`; matches inside a negation scope are
    dropped. Categories without the flag behave exactly as before.
    """
    results = []
    for cat in categories:
        neg_sensitive = cat.get("negation_sensitive", False)
        detected = []
        # Ordinary patterns count unconditionally; context-anchored patterns
        # (ambiguous asset terms) count only with a same-sentence money anchor.
        pattern_specs = (
            [(p, False) for p in cat["patterns"]]
            + [(p, True) for p in cat.get("context_anchored_patterns", [])]
        )
        for pattern, requires_money_context in pattern_specs:
            for m in pattern.finditer(text):
                if neg_sensitive and _is_negated(text, m.start()):
                    continue
                if requires_money_context and not _MONEY_CONTEXT.search(
                    _sentence_span(text, m.start())
                ):
                    continue
                snippet = m.group(0).strip()
                if snippet and snippet not in detected:
                    detected.append(snippet)
                if len(detected) >= 6:
                    break
            if len(detected) >= 6:
                break
        if detected:
            pct, words = _attribution_for_snippets(
                detected, lime_index, lime_total
            )
            results.append({
                "id": cat["id"],
                "title": cat["title"],
                "detected": detected,
                "explanation": cat["explanation"],
                "model_attribution_pct": pct,
                "model_words": words,
            })
    return results


def analyse_posting(text, lime_explanation=None):
    """
    Detect known scam and genuine patterns in the raw posting text, and
    report how much of the model's attribution fell on the detected words.

    Args:
        text: the raw posting text (or raw OCR output).
        lime_explanation: optional list of {"word", "weight"} from LIME.
            When supplied, each detected category reports the share of
            total model attribution that landed on its matched words.
            When absent, attribution figures are None.

    Detection is knowledge-based and never influences the prediction.
    Attribution figures are measured from the model.
    """
    t = _normalise(text)

    lime_index = {}
    lime_total = 0.0
    if lime_explanation:
        for item in lime_explanation:
            key = item["word"].lower()
            if key not in lime_index:
                lime_index[key] = item
        lime_total = sum(abs(i["weight"]) for i in lime_explanation)

    warning = _scan(t, WARNING_CATEGORIES, lime_index, lime_total)
    genuine = _scan(t, GENUINE_CATEGORIES, lime_index, lime_total)

    if warning:
        if len(warning) == 1:
            warning_summary = (
                "This posting contains known scam warning patterns in 1 category."
            )
        else:
            warning_summary = (
                f"This posting contains known scam warning patterns in "
                f"{len(warning)} categories."
            )
    else:
        warning_summary = (
            "No known scam warning patterns were found in this posting."
        )

    if genuine:
        if len(genuine) == 1:
            genuine_summary = (
                "1 indicator typical of legitimate job advertisements was found."
            )
        else:
            genuine_summary = (
                f"{len(genuine)} indicators typical of legitimate job "
                f"advertisements were found."
            )
    else:
        genuine_summary = (
            "No indicators typical of legitimate job advertisements were found."
        )

    return {
        "warning": warning,
        "genuine": genuine,
        "warning_summary": warning_summary,
        "genuine_summary": genuine_summary,
    }