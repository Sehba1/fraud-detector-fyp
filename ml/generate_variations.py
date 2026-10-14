"""
Generate training variations from collected real job postings.

This script expands a set of manually collected, real job postings into a
larger training set. Two kinds of variation are applied:

  1. Surface substitution: company names, monetary amounts, currencies,
     contact numbers, job titles, and locations are replaced from pools.

  2. Structural variation: sentence order is shuffled, sentences are
     randomly dropped, and some variations are short fragments rather
     than complete postings.

The second is important. An earlier version applied surface substitution
only, which left every variation of a given source sharing the same
sentence skeleton. The model learned that skeleton rather than the fraud
concept: LIME attributed decisions to filler words such as "dear" and
"delighted", which were constant across all variations of one source
letter. Structural variation removes those constants.

Source boosting

Some document formats are represented far more heavily on one side than
the other. Adding eight genuine offer letters caused the model to treat
offer-letter format itself as a genuine signal, so a fee-demanding offer
letter was classified genuine at 0.08%. BOOSTED_SOURCES raises the
variation count for the fraudulent examples sharing that format, so the
contrast between them is balanced and the model must attend to the
content rather than the format.

Method and limitations, for the project report:

  - This is deterministic template variation, not language generation.
    Every output is traceable to a specific real source via 'derived_from'.
  - Structural variation reduces but does not eliminate similarity between
    variations of the same source. Broader coverage still depends on the
    number of distinct real sources collected.
  - A held-out set of real examples is reserved and never used for
    training, so reported evaluation figures come from genuine data only.
  - A fixed random seed makes the output reproducible on a given machine.
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
VARIATIONS_PER_FRAUD = 18
VARIATIONS_PER_GENUINE = 18

# Sources whose document format is heavily represented on the opposing
# side need more variations to balance the contrast. The advance-fee
# offer letters are outnumbered by the genuine offer letters, which
# caused the model to treat offer-letter format as a genuine signal.
BOOSTED_SOURCES = {"scam_002", "scam_017"}
VARIATIONS_BOOSTED = 60

# Structural variation settings.
DROP_SENTENCE_PROB = 0.25      # chance a given sentence is dropped
SHUFFLE_PROB = 0.5             # chance sentence order is shuffled
FRAGMENT_PROB = 0.3            # chance the variation is a short fragment
FRAGMENT_MIN = 2               # minimum sentences in a fragment
FRAGMENT_MAX = 5               # maximum sentences in a fragment
MIN_SENTENCES_KEPT = 2         # never reduce a posting below this

# Real examples reserved for evaluation, never used to generate training
# data. Chosen to cover every scam type and a range of genuine styles.
HOLDOUT_IDS = {
    # fraudulent, one or two per scam type
    "scam_001",  # advance fee, the author's own case
    "scam_004",  # fake check, equipment purchase
    "scam_009",  # task scam
    "scam_014",  # brand impersonation, typosquatted domain
    "scam_016",  # regional Malaysian poster
    "scam_020",  # reshipping mule
    "scam_025",  # identity harvesting
    "scam_027",  # short Malaysian messaging redirect
    # genuine, covering the shapes the model finds hardest
    "real_003",  # contains "no prior experience required"
    "real_008",  # mentions passport requirement
    "real_009",  # very short Malaysian posting
    "real_012",  # contains "start immediately"
    "real_018",  # genuine posting containing anti-scam warning
    "real_024",  # no company name, no salary
}

# ── Substitution pools ───────────────────────────────────────────────────────

FAKE_COMPANIES = [
    "Vertexa Solutions", "Brightpath Innovations", "Nexora Digital",
    "Quantalis Group", "Orionex Technologies", "Silverline Ventures",
    "Meridian Global Services", "Apexon Consulting", "Everline Systems",
    "Northgate Holdings", "Crestwave Media", "Pinnacle Reach Corp",
    "Lumina Works", "Stratos Enterprise", "Halcyon Partners",
    "Zenith Bridge Ltd", "Cobalt Ridge Group", "Emberline Tech",
    "Novadyne Services", "Trueline Associates", "Kestrel Digital",
    "Arcvale Solutions", "Blueharbor Group", "Solstice Media Ltd",
    "Ironwood Consulting", "Vantage Reach Global", "Clearpoint Logistics",
]

IMPERSONATED_BRANDS = [
    "Meta Careers", "Amazon Talent Team", "Netflix Recruitment",
    "Adobe Hiring", "Salesforce Careers", "Cisco Talent",
    "Oracle Recruitment", "Dell Technologies Careers", "IBM Talent",
    "Shopify Hiring Team", "Spotify Recruitment", "Airbnb Careers",
    "Uber Talent Acquisition", "PayPal Hiring", "Intel Careers",
    "Samsung Careers", "Alibaba Recruitment", "Lazada Talent",
]

REAL_COMPANIES = [
    "Maybank", "CIMB Group", "Petronas Digital", "AirAsia",
    "Grab Malaysia", "Axiata Digital", "Sunway Group", "Gamuda Berhad",
    "Telekom Malaysia", "Sime Darby", "IOI Corporation", "YTL Corporation",
    "Hong Leong Bank", "Public Bank Berhad", "MR DIY Group",
    "Accenture Malaysia", "Deloitte Malaysia", "KPMG Malaysia",
    "Shopee Malaysia", "Lazada Malaysia", "Infosys", "Wipro Technologies",
    "Capgemini", "DXC Technology", "NTT Data", "Digi Telecommunications",
    "RHB Bank", "Top Glove Corporation",
]

SCAM_JOB_TITLES = [
    "Data Entry Clerk", "Online Task Assistant", "Remote Typist",
    "Content Promotion Assistant", "Product Reviewer",
    "Customer Service Associate", "Virtual Assistant",
    "Social Media Booster", "Order Processing Assistant",
    "Merchant Data Updater", "Live Chat Agent", "Transcription Assistant",
    "Package Forwarding Agent", "Payment Coordinator",
    "Personal Assistant", "Shipments Expert", "Sourcing Officer",
    "Chat Support Specialist", "Data Provider User",
]

REAL_JOB_TITLES = [
    "Software Engineer", "Data Analyst", "Backend Developer",
    "Machine Learning Engineer", "Business Analyst", "QA Engineer",
    "DevOps Engineer", "Product Manager", "Systems Administrator",
    "Frontend Developer", "Cloud Engineer", "Cybersecurity Analyst",
    "Database Administrator", "Technical Support Specialist",
    "Full Stack Developer", "Mobile Application Developer",
    "Network Engineer", "Data Engineer", "UI UX Designer",
    "Project Coordinator", "Accounting Assistant", "AI Developer",
]

PERSON_NAMES = [
    "Sarah Mitchell", "David Chen", "Amara Osei", "Priya Nair",
    "James Whitfield", "Lena Kowalski", "Marcus Bell", "Aisha Rahman",
    "Tomas Ruiz", "Hannah Park", "Daniel Okoro", "Mei Ling Tan",
    "Robert Ellis", "Fatima Zahra", "Kevin Brooks", "Nadia Petrova",
    "Samuel Adeyemi", "Grace Lim", "Oliver Hayes", "Ravi Kumar",
    "Elena Sokolova", "Adam Fitzgerald", "Yasmin Idris", "Peter Lau",
]

CURRENCIES = ["RM", "USD", "SGD", "INR", "PHP", "IDR", "THB"]

DAILY_AMOUNTS = ["150", "200", "250", "300", "400", "500", "600", "750", "800", "950"]
MONTHLY_AMOUNTS = ["3000", "4500", "5000", "6000", "7500", "8000", "9000", "12000"]
FEE_AMOUNTS = ["50", "99", "150", "250", "299", "500", "1200", "1700", "2500", "5000", "15000"]
HOURLY_AMOUNTS = ["28.50", "32.00", "35.75", "42.00", "45.50", "49.50", "55.00", "62.00"]

MESSAGING_APPS = ["WhatsApp", "Telegram", "WeChat", "Signal", "Viber", "LINE"]

PHONE_PREFIXES = ["+60", "+65", "+91", "+1", "+63", "+62", "+66"]

MY_LOCATIONS = [
    "Kuala Lumpur", "Petaling Jaya", "Shah Alam", "Cyberjaya", "Penang",
    "Johor Bahru", "Subang Jaya", "Puchong", "Cheras", "Ampang",
    "Bangsar", "Mont Kiara", "Klang", "Seremban", "Ipoh", "Bandar Utama",
]

INTL_LOCATIONS = [
    "Singapore", "Bangalore", "Manila", "Jakarta", "Bangkok",
    "Ho Chi Minh City", "London", "New York", "Sydney", "Toronto",
    "Dubai", "Hong Kong", "Mumbai", "Vientiane",
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


def _split_sentences(text):
    """
    Split into sentence-like units. Deliberately simple: postings are not
    well-formed prose, so a regex on terminal punctuation is adequate and
    avoids a dependency.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _restructure(text):
    """
    Apply structural variation: optionally take a fragment, drop random
    sentences, and optionally shuffle the order.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= MIN_SENTENCES_KEPT:
        return text

    if random.random() < FRAGMENT_PROB and len(sentences) > FRAGMENT_MAX:
        n = random.randint(FRAGMENT_MIN, FRAGMENT_MAX)
        start = random.randint(0, len(sentences) - n)
        sentences = sentences[start:start + n]
    else:
        kept = [s for s in sentences if random.random() > DROP_SENTENCE_PROB]
        if len(kept) >= MIN_SENTENCES_KEPT:
            sentences = kept

    if random.random() < SHUFFLE_PROB:
        random.shuffle(sentences)

    return " ".join(sentences)


def _substitute(text, label, scam_type):
    """
    Replace surface details that genuinely vary between real instances of
    the same posting type.
    """
    out = text

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
        "Horizon NJ Health", "Cargo Group Logistics", "General Efforts Foundation",
        "HARBIN PHARMACEUTICAL GROUP", "TEMU Services USA", "TEMU",
        "BrightPath Solutions", "Mara", "Confido", "Revature", "8nabler",
        "Lenovo", "FedEx", "SEAhub Asia", "TikTok Shop", "Google",
        "Microsoft", "Amazon Web Services Malaysia SDN BHD", "EY",
        "Crowe KL Tax Sdn Bhd", "Bybit", "Deloitte", "SLB", "Worldline",
        "MoneyLion", "Maxis", "Noventiq Solutions International Sdn Bhd",
        "Noventiq", "Siemens Healthineers", "Setel", "Amazon",
        "Barcia Auditing Firm Ltd", "Nitro Branding Inc", "Richmond College",
        "Emertxe Information Technologies", "Emertxe", "Hex Softwares Pvt Ltd",
        "Hex Softwares", "Wipro Ltd", "CodeAlpha", "CODE ALPHA", "LTM",
    ]
    for name in known_names:
        out = out.replace(name, company)

    known_people = [
        "P. Anthony Reddy", "John Sutton", "Quan Nguyen", "Kathleen Dwyer",
        "Jasmine Martine", "Seraphina", "Alexa", "Lacy", "Ankur",
        "Sehba Hanief", "Nitin Singh", "Shirley", "Oscar Steinmann",
        "David Johnston", "Sarah", "Sophia", "Sonam Raj", "Robert Tyrell",
        "David Bartholomew", "Angela Griesmann", "Eugene Woo",
        "Helena Waxler", "Nikhilaa Sivakumar", "Mubeen Jukaku",
        "Jayakmar Balasubramanian", "Akeem Tunde", "Divya Tripathi",
        "Life Okokon Ebuk", "Swati Srivastava",
    ]
    for person in known_people:
        out = out.replace(person, _pick(PERSON_NAMES))

    title_pool = SCAM_JOB_TITLES if label == "fraudulent" else REAL_JOB_TITLES
    known_titles = [
        "Python Developer Internship Program", "Customer Service Associate",
        "Quality Engineers", "Transcriptionist", "Personal Assistant",
        "Senior Marketing Manager", "Administrative Assistant",
        "Remote Data Entry Live Chat Agent", "Payment Processing Specialist",
        "TELEMARKETER", "content promotion assistant", "Shipments Expert",
        "Sourcing Officer", "Online Chat Support Specialist",
        "Data Provider User", "Python Developer", "Accounts Manager",
        "Marketing Strategist",
    ]
    for t in known_titles:
        out = out.replace(t, _pick(title_pool))

    # Monetary amounts.
    out = re.sub(r"\b25,?000\b", _pick(FEE_AMOUNTS), out)
    out = re.sub(r"\b15,?000\b", _pick(FEE_AMOUNTS), out)
    out = re.sub(r"\b10,?000\b", _pick(MONTHLY_AMOUNTS), out)
    out = re.sub(r"\b299\b", _pick(FEE_AMOUNTS), out)
    out = re.sub(r"\b1700\b", _pick(FEE_AMOUNTS), out)
    out = re.sub(r"\b6,?300\b", _pick(MONTHLY_AMOUNTS), out)
    out = re.sub(r"\b3850\b", _pick(MONTHLY_AMOUNTS), out)
    out = re.sub(r"\b49\.50\b", _pick(HOURLY_AMOUNTS), out)
    out = re.sub(r"\b35\.00\b", _pick(HOURLY_AMOUNTS), out)
    out = re.sub(r"\b55,?000\b", _pick(MONTHLY_AMOUNTS), out)
    out = re.sub(r"\b85,?000\b", _pick(MONTHLY_AMOUNTS), out)
    out = re.sub(r"\bRM1450\b", f"RM{_pick(MONTHLY_AMOUNTS)}", out)
    out = re.sub(r"\bRM1550\b", f"RM{_pick(MONTHLY_AMOUNTS)}", out)
    out = re.sub(r"\b(\d+) dollars per day\b",
                 lambda m: f"{_pick(DAILY_AMOUNTS)} dollars per day", out)
    out = re.sub(r"\b500 dollars\b", f"{_pick(DAILY_AMOUNTS)} dollars", out)
    out = re.sub(r"\bRm3000-3500\b", f"RM{_pick(MONTHLY_AMOUNTS)}", out)
    out = re.sub(r"\bRm6000-7000\b", f"RM{_pick(MONTHLY_AMOUNTS)}", out)

    out = re.sub(r"\bRs\.?\s?", f"{_pick(CURRENCIES)} ", out)

    for app in ["WhatsApp", "WhatsAPP", "Telegram", "WeChat"]:
        if app in out:
            out = out.replace(app, _pick(MESSAGING_APPS))

    out = re.sub(r"\+\d[\d\-\s]{7,}", _phone(), out)

    for loc in ["Kuala Lumpur", "Cheras Alam Damai", "New York City",
                "Bandar Utama", "Petaling Jaya", "Bangalore"]:
        if loc in out:
            pool = MY_LOCATIONS if label == "genuine" else MY_LOCATIONS + INTL_LOCATIONS
            out = out.replace(loc, _pick(pool))

    if label == "genuine":
        out = out.replace("Computer Science", _pick(DEGREE_FIELDS), 1)

    if label == "genuine" and random.random() < 0.4:
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

    missing = HOLDOUT_IDS - {r["id"] for r in rows}
    if missing:
        raise SystemExit(f"Holdout ids not present in input: {sorted(missing)}")

    holdout = [r for r in rows if r["id"] in HOLDOUT_IDS]
    sources = [r for r in rows if r["id"] not in HOLDOUT_IDS]

    generated = []
    boosted_used = []
    for src in sources:
        if src["id"] in BOOSTED_SOURCES:
            n = VARIATIONS_BOOSTED
            boosted_used.append(src["id"])
        else:
            n = (VARIATIONS_PER_FRAUD if src["label"] == "fraudulent"
                 else VARIATIONS_PER_GENUINE)

        seen = set()
        for i in range(n):
            text = _restructure(src["text"])
            text = _substitute(text, src["label"], src.get("scam_type", ""))
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

    # Source examples themselves are included, marked as real.
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
    print(f"Boosted sources:           {sorted(boosted_used) or 'none found'}")
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