from app.core.patterns import analyse_posting

scam = "URGENT HIRING! Work from home. No experience needed. Earn RM500 per day. Send bank account details and passport copy. Contact us on WhatsApp. Registration fee of RM50 required."
genuine = "Software Engineer, Backend. About us: an established firm. Responsibilities include maintaining services. Requirements: Bachelor degree, 3 years of professional experience with Python. We offer medical and dental insurance, EPF and SOCSO contributions, annual leave. Apply through our careers portal. Shortlisted candidates complete a technical assessment and interview."

for name, text in [("SCAM", scam), ("GENUINE", genuine)]:
    r = analyse_posting(text)
    s = r["risk_score"]
    print("=" * 55)
    print(name, "-> score", s["score"], "/100 |", s["band_label"])
    print("  warning points:", s["warning_points"], " genuine points:", s["genuine_points"], " raw:", s["raw_total"])
    for it in s["items"]:
        sign = "+" if it["direction"] == "add" else "-"
        print("   ", sign + str(it["points"]), it["title"])
    print()
