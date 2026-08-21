"""
Generate training variations from collected real job postings.

This script expands a small set of manually collected, real job postings
into a larger training set by substituting surface details: company names,
monetary amounts, currencies, contact numbers, job titles, and locations.

Sentence structure and rhetorical strategy are preserved, because those
are the properties the model needs to learn. Only the details that vary
between real instances of the same scam are substituted.

Method and limitations, for the project report:

  - This is deterministic template substitution, not language generation.
    Every output is traceable to a specific real source example via the
    'derived_from' field.
  - Because structure is preserved, variations of one source example are
    structurally similar to each other. The model may therefore learn
    template shape in addition to the underlying concept. This is a
    stated limitation of the augmentation approach.
  - A held-out set of real examples is reserved and never used for
    training, so reported evaluation figures come from genuine data only.
  - A fixed random seed makes the output reproducible.

Usage:
    python ml/generate_variations.py
"""

import json
import random
import re
from pathlib import Path

SEED = 42
random.seed(SEED)

# Paths relative to project root
INPUT_PATH = Path("ml/data/raw/collected/real_postings.jsonl")
OUTPUT_DIR = Path("ml/data/raw/collected")
TRAIN_OUT = OUTPUT_DIR / "augmented_train.jsonl"
HOLDOUT_OUT = OUTPUT_DIR / "holdout_real.jsonl"

# How many variations to generate per source example.
VARIATIONS_PER_FRAUD = 25
VARIATIONS_PER_GENUINE = 28

# Real examples reserved for evaluation and never used to generate
# training data. Chosen to cover distinct scam types plus a range of
# genuine posting styles.
HOLDOUT_IDS = {
    "scam_001",  # advance fee, the author's own case
    "scam_004",  # fake check, equipment purchase
    "scam_009",  # task scam, brand impersonation
    "scam_011",  # messaging redirect
    "scam_014",  # brand impersonation, typosquatted domain
    "scam_016",  # regional Malaysian scam
    "real_004",  # Malaysian startup, informal contact
    "real_008",  # Malaysian corporate, mentions passport
    "real_009",  # very short Malaysian posting
    "real_012",  # contains "start immediately"
    "real_003",  # contains "no prior experience required"
}

# ── Substitution pools ───────────────────────────────────────────────────────
# Values are drawn from these pools to replace surface details. Pools are
# deliberately varied in register and region so the model does not learn a
# narrow vocabulary.

FAKE_COMPANIES = [
    "Vertexa Solutions", "Brightpath Innovations", "Nexora Digital",
    "Quantalis Group", "Orionex Technologies", "Silverline Ventures",
    "Meridian Global Services", "Apexon Consulting", "Everline Systems",
    "Northgate Holdings", "Crestwave Media", "Pinnacle Reach Corp",
    "Lumina Works", "Stratos Enterprise", "Halcyon Partners",
    "Zenith Bridge Ltd", "Cobalt Ridge Group", "Emberline Tech",
    "Novadyne Services", "Trueline Associates", "Kestrel Digital",
    "Arcvale Solutions", "Blueharbor Group", "Solstice Media Ltd",
    "Ironwood Consulting",
]

IMPERSONATED_BRANDS = [
    "Meta Careers", "Amazon Talent Team", "Netflix Recruitment",
    "Adobe Hiring", "Salesforce Careers", "Cisco Talent",
    "Oracle Recruitment", "Dell Technologies Careers", "IBM Talent",
    "Shopify Hiring Team", "Spotify Recruitment", "Airbnb Careers",
    "Uber Talent Acquisition", "PayPal Hiring", "Intel Careers",
]

REAL_COMPANIES = [
    "Maybank", "CIMB Group", "Petronas Digital", "AirAsia",
    "Grab Malaysia", "Axiata Digital", "Sunway Group", "Gamuda Berhad",
    "Telekom Malaysia", "Sime Darby", "IOI Corporation", "YTL Corporation",
    "Hong Leong Bank", "Public Bank Berhad", "MR DIY Group",
    "Accenture Malaysia", "Deloitte Malaysia", "KPMG Malaysia",
    "Shopee Malaysia", "Lazada Malaysia", "Infosys", "Wipro Technologies",
    "Capgemini", "DXC Technology", "NTT Data",
]

SCAM_JOB_TITLES = [
    "Data Entry Clerk", "Online Task Assistant", "Remote Typist",
    "Content Promotion Assistant", "Product Reviewer",
    "Customer Service Associate", "Virtual Assistant",
    "Social Media Booster", "Order Processing Assistant",
    "Merchant Data Updater", "Live Chat Agent", "Transcription Assistant",
    "Package Forwarding Agent", "Payment Coordinator",
    "Personal Assistant",
]

REAL_JOB_TITLES = [
    "Software Engineer", "Data Analyst", "Backend Developer",
    "Machine Learning Engineer", "Business Analyst", "QA Engineer",
    "DevOps Engineer", "Product Manager", "Systems Administrator",
    "Frontend Developer", "Cloud Engineer", "Cybersecurity Analyst",
    "Database Administrator", "Technical Support Specialist",
    "Full Stack Developer", "Mobile Application Developer",
    "Network Engineer", "Data Engineer", "UI UX Designer",
    "Project Coordinator",
]

PERSON_NAMES = [
    "Sarah Mitchell", "David Chen", "Amara Osei", "Priya Nair",
    "James Whitfield", "Lena Kowalski", "Marcus Bell", "Aisha Rahman",
    "Tomas Ruiz", "Hannah Park", "Daniel Okoro", "Mei Ling Tan",
    "Robert Ellis", "Fatima Zahra", "Kevin Brooks", "Nadia Petrova",
    "Samuel Adeyemi", "Grace Lim", "Oliver Hayes", "Ravi Kumar",
]

CURRENCIES = ["RM", "USD", "SGD", "INR", "PHP", "IDR"]

DAILY_AMOUNTS = ["150", "200", "250", "300", "400", "500", "600", "750", "800"]
MONTHLY_AMOUNTS = ["3000", "4500", "5000", "6000", "7500", "8000", "9000", "12000"]
FEE_AMOUNTS = ["50", "99", "150", "250", "299", "500", "1200", "2500", "5000", "15000"]
HOURLY_AMOUNTS = ["28.50", "32.00", "35.75", "42.00", "45.50", "49.50", "55.00", "62.00"]

MESSAGING_APPS = ["WhatsApp", "Telegram", "WeChat", "Signal", "Viber", "LINE"]

PHONE_PREFIXES = ["+60", "+65", "+91", "+1", "+63", "+62"]

MY_LOCATIONS = [
    "Kuala Lumpur", "Petaling Jaya", "Shah Alam", "Cyberjaya", "Penang",
    "Johor Bahru", "Subang Jaya", "Puchong", "Cheras", "Ampang",
    "Bangsar", "Mont Kiara", "Klang", "Seremban", "Ipoh",
]

INTL_LOCATIONS = [
    "Singapore", "Bangalore", "Manila", "Jakarta", "Bangkok", "Ho Chi Minh City",
    "London", "New York", "Sydney", "Toronto", "Dubai", "Hong Kong",
]

DEGREE_FIELDS = [
    "Computer Science", "Information Technology", "Software Engineering",
    "Data Science", "Business Administration", "Accounting", "Marketing",
    "Electrical Engineering", "Information Systems", "Finance",
]

BENEFIT_SETS = [
    "EPF and SOCSO contributions, medical insurance, and annual leave",
    "health, dental and vision coverage, retirement contributions, and paid time off",
    "medical and dental insurance, EPF, SOCSO, and performance bonuses",
    "annual leave, medical coverage, statutory contributions, and training allowance",
    "comprehensive health insurance, provident fund, and 18 days annual leave",
]

PROCESS_SETS = [
    "Shortlisted candidates will be contacted for a technical assessment followed by two interview rounds.",
    "The hiring process consists of a resume screening, an online assessment, and a panel interview.",
    "Selected applicants will be invited for an initial phone screening and a face to face interview.",
    "Candidates will complete a written assessment before proceeding to interviews with the hiring team.",
    "Our process includes an application review, a technical interview, and a final discussion with the department head.",
]


def _pick(pool):
    return random.choice(pool)


def _phone():
    prefix = _pick(PHONE_PREFIXES)
    digits = "".join(str(random.randint(0, 9)) for _ in range(9))
    return f"{prefix}{digits}"


def _substitute(text, label, scam_type):
    """
    Apply surface substitutions to one posting.

    Replacements target details that genuinely vary between real instances
    of the same posting type. Structure and rhetorical strategy are left
    intact.
    """
    out = text

    # Company names, chosen from a pool appropriate to the label.
    if label == "fraudulent":
        if scam_type == "brand_impersonation":
            company = _pick(IMPERSONATED_BRANDS)
        else:
            company = _pick(FAKE_COMPANIES)
    else:
        company = _pick(REAL_COMPANIES)

    known_names = [
        "Intigrityfactor Innovations", "Intigrityfactor", "GNI Aviation Services",
        "Numeric Technologies", "PROPIO", "FxPro", "WBD Global Streaming",
        "TOP TALENT PLANET CORP", "DigiHub", "Oculus", "L3Harris Technologies",
        "Horizon NJ Health", "Confido", "Revature", "8nabler", "Lenovo",
        "FedEx", "SEAhub Asia", "TikTok Shop", "Google", "Microsoft",
        "Amazon Web Services Malaysia SDN BHD", "EY",
    ]
    for name in known_names:
        out = out.replace(name, company)

    # Person names in signatures and introductions.
    known_people = [
        "P. Anthony Reddy", "John Sutton", "Quan Nguyen", "Kathleen Dwyer",
        "Jasmine Martine", "Seraphina", "Alexa", "Lacy", "Ankur",
        "Sehba Hanief",
    ]
    for person in known_people:
        out = out.replace(person, _pick(PERSON_NAMES))

    # Job titles.
    title_pool = SCAM_JOB_TITLES if label == "fraudulent" else REAL_JOB_TITLES
    known_titles = [
        "Python Developer Internship Program", "Customer Service Associate",
        "Quality Engineers", "Transcriptionist", "Personal Assistant",
        "Senior Marketing Manager", "Administrative Assistant",
        "Remote Data Entry Live Chat Agent", "Payment Processing Specialist",
        "TELEMARKETER", "content promotion assistant",
    ]
    for t in known_titles:
        out = out.replace(t, _pick(title_pool))

    # Monetary amounts. Order matters: longer patterns first.
    out = re.sub(r"\b25,000\b", _pick(FEE_AMOUNTS), out)
    out = re.sub(r"\b15,000\b", _pick(FEE_AMOUNTS), out)
    out = re.sub(r"\b10,000\b", _pick(MONTHLY_AMOUNTS), out)
    out = re.sub(r"\b299\b", _pick(FEE_AMOUNTS), out)
    out = re.sub(r"\b6,300\b", _pick(MONTHLY_AMOUNTS), out)
    out = re.sub(r"\b49\.50\b", _pick(HOURLY_AMOUNTS), out)
    out = re.sub(r"\b35\.00\b", _pick(HOURLY_AMOUNTS), out)
    out = re.sub(r"\b75 dollars per hour\b",
                 f"{_pick(HOURLY_AMOUNTS)} dollars per hour", out)
    out = re.sub(r"\b500 dollars\b", f"{_pick(DAILY_AMOUNTS)} dollars", out)
    out = re.sub(r"\bRm3000-3500\b",
                 f"RM{_pick(MONTHLY_AMOUNTS)}", out)
    out = re.sub(r"\bRm6000-7000\b",
                 f"RM{_pick(MONTHLY_AMOUNTS)}", out)

    # Currency markers.
    out = re.sub(r"\bRs\.\s?", f"{_pick(CURRENCIES)} ", out)

    # Messaging apps.
    for app in ["WhatsApp", "Telegram", "WeChat"]:
        if app in out:
            out = out.replace(app, _pick(MESSAGING_APPS))

    # Phone numbers.
    out = re.sub(r"\+\d[\d\-\s]{7,}", _phone(), out)

    # Locations.
    for loc in ["Kuala Lumpur", "Cheras Alam Damai", "New York NY", "New York City"]:
        if loc in out:
            pool = MY_LOCATIONS if label == "genuine" else MY_LOCATIONS + INTL_LOCATIONS
            out = out.replace(loc, _pick(pool))

    # Degree fields, genuine postings only.
    if label == "genuine":
        out = out.replace("Computer Science", _pick(DEGREE_FIELDS), 1)

    # Benefit and process phrasing, genuine postings only. Adds variety in
    # the sections that most distinguish real postings.
    if label == "genuine" and random.random() < 0.5:
        out += " We offer " + _pick(BENEFIT_SETS) + ". " + _pick(PROCESS_SETS)

    return out


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    rows = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    holdout = [r for r in rows if r["id"] in HOLDOUT_IDS]
    sources = [r for r in rows if r["id"] not in HOLDOUT_IDS]

    missing = HOLDOUT_IDS - {r["id"] for r in rows}
    if missing:
        raise SystemExit(f"Holdout ids not present in input: {sorted(missing)}")

    generated = []
    for src in sources:
        n = (VARIATIONS_PER_FRAUD if src["label"] == "fraudulent"
             else VARIATIONS_PER_GENUINE)
        seen = set()
        for i in range(n):
            text = _substitute(src["text"], src["label"], src.get("scam_type", ""))
            # Skip exact duplicates so the set stays varied.
            if text in seen:
                continue
            seen.add(text)
            generated.append({
                "id": f"{src['id']}_v{i:03d}",
                "label": src["label"],
                "scam_type": src.get("scam_type", ""),
                "derived_from": src["id"],
                "synthetic": True,
                "text": text,
            })

    # Source examples themselves are included in training, marked as real.
    for src in sources:
        generated.append({
            "id": src["id"],
            "label": src["label"],
            "scam_type": src.get("scam_type", ""),
            "derived_from": src["id"],
            "synthetic": False,
            "text": src["text"],
        })

    random.shuffle(generated)

    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for row in generated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with HOLDOUT_OUT.open("w", encoding="utf-8") as f:
        for row in holdout:
            row = dict(row)
            row["synthetic"] = False
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_fraud = sum(1 for r in generated if r["label"] == "fraudulent")
    n_genuine = sum(1 for r in generated if r["label"] == "genuine")
    n_real = sum(1 for r in generated if not r["synthetic"])

    print(f"Source examples used:      {len(sources)}")
    print(f"Held out for evaluation:   {len(holdout)}")
    print()
    print(f"Training set written to:   {TRAIN_OUT}")
    print(f"  total:                   {len(generated)}")
    print(f"  fraudulent:              {n_fraud}")
    print(f"  genuine:                 {n_genuine}")
    print(f"  real (not synthetic):    {n_real}")
    print()
    print(f"Holdout set written to:    {HOLDOUT_OUT}")
    print(f"  total:                   {len(holdout)}")
    print(f"  fraudulent:              {sum(1 for r in holdout if r['label'] == 'fraudulent')}")
    print(f"  genuine:                 {sum(1 for r in holdout if r['label'] == 'genuine')}")


if __name__ == "__main__":
    main()