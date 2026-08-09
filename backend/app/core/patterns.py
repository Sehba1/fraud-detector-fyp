"""
Rule-based scam pattern analysis for the Fraud Detector backend.

This module is the knowledge-based half of the system's hybrid explanation
framework. It scans the raw posting text for phrases and patterns that are
documented indicators of recruitment fraud, and for indicators typical of
legitimate job advertisements.

Important design facts:

  - This layer is completely independent of the BERT model and of LIME.
    It never influences the prediction, the confidence, or the risk tier.
  - It runs on the RAW input text (or raw OCR output), lightly normalised
    to lowercase with collapsed whitespace. It does not use the BERT
    cleaning pipeline, because that pipeline strips digits and punctuation
    and would destroy patterns like "RM500/day" or phone numbers.
  - Every category shown to the user carries the actual matched text
    snippets as evidence, so nothing is asserted without proof from the
    posting itself.
  - Negation handling: the layer performs surface pattern matching and
    does not understand negation in general. High-frequency negated
    phrases ("no experience required", "no interview", "do not apply
    through") are explicitly guarded with negative lookbehinds so they do
    not fire as genuine signals. The general limitation is documented in
    the project report.
  - The pattern list is a maintained knowledge base. It is expected to
    grow as new scam variants are encountered; advance-fee "membership
    fee" offer-letter scams were added after one was observed in testing.
"""

import re


def _c(pattern: str):
    return re.compile(pattern)


# ── Warning pattern categories ────────────────────────────────────────────────
WARNING_CATEGORIES = [
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
        "id": "low_requirements",
        "title": "Suspiciously low entry requirements",
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
        "id": "money_requests",
        "title": "Requests for money or financial details",
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
            _c(r"\bbitcoin\b"),
            _c(r"\busdt\b"),
            _c(r"\bcrypto(?:currency)?\b"),
            _c(r"\binvestment (?:opportunity|plan|scheme|return)\b"),
        ],
    },
    {
        "id": "identity_requests",
        "title": "Personal identity information requests",
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
]


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace. Keeps digits and punctuation."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _scan(text: str, categories) -> list:
    """
    Run every category's patterns against the text. A category is included
    only if at least one pattern matched, and it carries the actual matched
    snippets (deduplicated, capped at 6) as evidence.
    """
    results = []
    for cat in categories:
        detected = []
        for pattern in cat["patterns"]:
            for m in pattern.finditer(text):
                snippet = m.group(0).strip()
                if snippet and snippet not in detected:
                    detected.append(snippet)
                if len(detected) >= 6:
                    break
            if len(detected) >= 6:
                break
        if detected:
            results.append({
                "id": cat["id"],
                "title": cat["title"],
                "detected": detected,
                "explanation": cat["explanation"],
            })
    return results


def analyse_posting(text: str) -> dict:
    """
    Analyse the raw posting text for known scam and genuine patterns.

    Returns a dictionary with warning categories, genuine categories, and
    templated summary sentences for each side. Completely independent of
    the model prediction.
    """
    t = _normalise(text)
    warning = _scan(t, WARNING_CATEGORIES)
    genuine = _scan(t, GENUINE_CATEGORIES)

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