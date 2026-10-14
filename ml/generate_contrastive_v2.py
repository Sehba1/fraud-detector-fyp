"""
Generate contrastive training pairs.

Motivation, for the project report:

Occlusion analysis on the previous model showed that removing every
fee-related sentence from a known advance-fee scam letter changed the
fraud probability by 0.002 percentage points. The model was classifying
the letter correctly, but not because of the fee: it was separating the
classes on tone, register, and structure instead. Explanations therefore
attributed the decision to filler words such as "dear" and "delighted",
which was an accurate report of shallow reasoning.

The cause is that in the existing training data, fraudulent and genuine
postings differ in many ways simultaneously. The model can separate them
using whichever cue is easiest, and never has to learn that a fee demand
matters.

This script removes that shortcut. For each genuine posting it produces
fraudulent twins that are identical except for one injected sentence
carrying a single scam signal. The label differs; almost nothing else
does. The injected signal becomes the only available discriminator, so
the model has to learn it.

Method notes:

  - Injected sentences are written from the phrasing observed in the
    collected real scams, not invented freely.
  - Each twin carries exactly one signal, so the contrast is clean.
  - The genuine originals are emitted alongside their twins, so the model
    sees both sides of every pair.
  - Injection position varies (start, middle, end) so position does not
    become a cue in itself.
  - Output is traceable: every row records its source and injected signal.

Usage:
    python ml/generate_contrastive.py
"""

import json
import random
import re
from pathlib import Path

SEED = 43  # differs from generate_variations.py so the two do not correlate
random.seed(SEED)

INPUT_PATH = Path("ml/data/raw/collected/real_postings.jsonl")
OUTPUT_PATH = Path("ml/data/raw/collected/contrastive_train.jsonl")

# Genuine sources reserved for evaluation are skipped here too, so held-out
# postings never leak into training in any form.
EXCLUDE_IDS = {
    "real_003", "real_008", "real_009", "real_012", "real_018", "real_024",
}

# How many distinct twins to generate per genuine source. Signals are
# sampled without replacement, so each twin carries a different signal.
TWINS_PER_SOURCE = 5

PRIORITY_SIGNALS = {
    "urgency": 4,
    "targeting": 4,
    "no_interview": 4,
    "unrealistic_earnings": 3,
    "fee_demand": 3,
    "fake_check": 2,
}
GENUINE_REPEAT = 6

# ── Scam signals ──────────────────────────────────────────────────────────────
# Each signal supplies several phrasings, drawn from the language observed
# in the collected real scams. Placeholders are filled at generation time.

SIGNALS = [
    {
        "id": "fee_demand",
        "title": "Fee or payment demand",
        "templates": [
            "To confirm your position, you are required to pay a one-time registration fee of {cur}{fee}.",
            "A membership fee of {cur}{fee} must be paid before your onboarding can be completed.",
            "Please note that a processing fee of {cur}{fee} is payable within 7 days of receiving this letter.",
            "Candidates are required to deposit a training fee of {cur}{fee} prior to commencement.",
            "Kindly complete the payment of {cur}{fee} using the link provided to activate your account.",
        ],
    },
    {
        "id": "messaging_contact",
        "title": "Contact through messaging apps",
        "templates": [
            "If you are interested, please add our hiring manager on {app} at {phone} to proceed.",
            "Kindly send your resume directly to {phone} on {app} rather than replying to this message.",
            "All further communication will take place through {app}. Message {phone} to begin.",
            "Do not reply here. Reach out to our recruiter on {app} at {phone} for the next steps.",
            "Please download {app} and add our manager to confirm your interview slot.",
        ],
    },
    {
        "id": "identity_request",
        "title": "Identity document request",
        "templates": [
            "Please send a clear copy of your passport and national identity card to complete registration.",
            "Before we proceed, kindly attach a photograph of your government-issued photo ID, front and back.",
            "You must send your full name, home address, phone number and a copy of your identity card right away.",
            "To register you in our system, please provide a scanned copy of your identity document and proof of address.",
            "Applicants are required to submit their passport copy before the interview can be scheduled.",
        ],
    },
    {
        "id": "unrealistic_earnings",
        "title": "Unrealistic earnings claim",
        "templates": [
            "You can earn between {cur}{low} and {cur}{high} per day working only 60 to 90 minutes.",
            "Daily payment of {cur}{low} to {cur}{high} is settled every evening directly to your account.",
            "Monthly income can reach {cur}{high} and above if you perform well.",
            "This role offers {cur}{low} per day with no prior experience required.",
            "Weekly payment of {cur}{high} is guaranteed regardless of workload.",
        ],
    },
    {
        "id": "urgency",
        "title": "Urgency and pressure",
        "templates": [
            "Limited positions are available and will be filled on a first come first served basis.",
            "You must respond within 24 hours or your application will be removed from consideration.",
            "Apply today. Slots are closing shortly and we cannot hold your place.",
            "Immediate joining required. Please confirm tonight to secure your position.",
            "Only 20 spots remain. Reply now to avoid missing this opportunity.",
        ],
    },
    {
        "id": "fake_check",
        "title": "Advance payment for equipment",
        "templates": [
            "Before you begin, a check will be issued to cover the purchase of your home office equipment.",
            "You will receive a payment which is to be used to buy the required equipment from our approved vendor.",
            "Funds for your workstation will be sent in advance. Purchase the listed items and return any balance.",
            "A corporate check will be mailed to you to cover the cost of the standard equipment package.",
            "Company funds will be transferred to your account so you can acquire the necessary software and hardware.",
        ],
    },
    {
        "id": "no_interview",
        "title": "Hiring without assessment",
        "templates": [
            "No interview is required. You have already been selected based on your profile.",
            "There is no assessment stage. Confirm your acceptance and training will begin immediately.",
            "No experience or qualifications are needed and anyone can apply.",
            "We are hiring without interviews this month. Simply reply to accept the position.",
            "Your application has been approved without further screening.",
        ],
    },
    {
        "id": "targeting",
        "title": "Discriminatory targeting",
        "templates": [
            "Female candidates only. Applicants must be under 30 years of age.",
            "This position is open to male applicants aged 18 to 28 only.",
            "Students and housewives are especially welcome to apply for this role.",
            "Only candidates aged below 25 will be considered for this position.",
        ],
    },
    {
        "id": "reshipping",
        "title": "Package handling duties",
        "templates": [
            "Your duties include receiving packages at your home address and forwarding them to our clients.",
            "You will be paid {cur}{low} per package processed and shipped from your residence.",
            "Incoming parcels will be routed to your address for repackaging and onward delivery.",
            "The role involves accepting deliveries on our behalf and posting them to overseas partners.",
        ],
    },
]

CURRENCIES = ["RM", "USD ", "SGD ", "INR ", "$"]
FEE_AMOUNTS = ["50", "99", "150", "250", "299", "500", "1,200", "2,500", "5,000"]
LOW_AMOUNTS = ["100", "150", "200", "250", "300", "400"]
HIGH_AMOUNTS = ["500", "800", "1,000", "2,000", "3,000", "6,000"]
APPS = ["WhatsApp", "Telegram", "WeChat", "Signal", "Viber"]
PHONE_PREFIXES = ["+60", "+65", "+91", "+1", "+63"]


def _phone():
    prefix = random.choice(PHONE_PREFIXES)
    digits = "".join(str(random.randint(0, 9)) for _ in range(9))
    return f"{prefix}{digits}"


def _fill(template):
    """Fill placeholders in an injection template."""
    return template.format(
        cur=random.choice(CURRENCIES),
        fee=random.choice(FEE_AMOUNTS),
        low=random.choice(LOW_AMOUNTS),
        high=random.choice(HIGH_AMOUNTS),
        app=random.choice(APPS),
        phone=_phone(),
    )


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def _inject(text, sentence):
    """
    Insert the sentence at the start, somewhere in the middle, or at the
    end. Varying position prevents position itself becoming the cue.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return text.rstrip() + " " + sentence

    where = random.random()
    if where < 0.2:
        pos = 1                       # after the opening line
    elif where < 0.8:
        pos = random.randint(1, len(sentences) - 1)
    else:
        pos = len(sentences)          # at the end

    sentences.insert(pos, sentence)
    return " ".join(sentences)


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    rows = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    genuine = [
        r for r in rows
        if r["label"] == "genuine" and r["id"] not in EXCLUDE_IDS
    ]

    if not genuine:
        raise SystemExit("No genuine sources available after exclusions.")

    out = []
    signal_counts = {s["id"]: 0 for s in SIGNALS}

    for src in genuine:
        for rep in range(GENUINE_REPEAT):
            out.append({
                "id": f"{src['id']}_pair_base_{rep}",
                "label": "genuine",
                "scam_type": "",
                "derived_from": src["id"],
                "injected_signal": None,
                "synthetic": True,
                "contrastive": True,
                "text": src["text"],
            })

        for signal in SIGNALS:
            copies = 1 + PRIORITY_SIGNALS.get(signal["id"], 0)
            used = set()
            for c in range(copies):
                choices = [t for t in signal["templates"] if t not in used] or signal["templates"]
                template = random.choice(choices)
                used.add(template)
                sentence = _fill(template)
                twin = _inject(src["text"], sentence)
                out.append({
                    "id": f"{src['id']}_pair_{signal['id']}_{c}",
                    "label": "fraudulent",
                    "scam_type": signal["id"],
                    "derived_from": src["id"],
                    "injected_signal": signal["id"],
                    "injected_sentence": sentence,
                    "synthetic": True,
                    "contrastive": True,
                    "text": twin,
                })
                signal_counts[signal["id"]] += 1

    random.shuffle(out)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_fraud = sum(1 for r in out if r["label"] == "fraudulent")
    n_genuine = sum(1 for r in out if r["label"] == "genuine")

    print(f"Genuine sources used:      {len(genuine)}")
    print(f"Twins per source:          {TWINS_PER_SOURCE}")
    print()
    print(f"Written to:                {OUTPUT_PATH}")
    print(f"  total:                   {len(out)}")
    print(f"  fraudulent (twins):      {n_fraud}")
    print(f"  genuine (originals):     {n_genuine}")
    print()
    print("Signal coverage:")
    for sid, count in sorted(signal_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {sid:<22} {count}")


if __name__ == "__main__":
    main()