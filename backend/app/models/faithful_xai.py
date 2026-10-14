"""
faithful_xai.py  --  production model-faithful XAI for the frozen V3 classifier.

Interventional group-Shapley on V3 against 10 genuine backgrounds; reference +
factor contributions = V3 score. Backgrounds cached once; subset scoring batched
and memoized; single-background evidence pass; units beyond the 256-token window
are not scored. V3 weights, the prediction, max_length=256, the reference method,
the factor definitions and the reconstruction math are all UNCHANGED.

NOTE ON HONESTY: the FACTORS below are PREDEFINED keyword groups used only to
partition the input for attribution. The trained model did NOT learn these named
categories. Each flagged factor now also reports `keywords` = the category-defining
terms that actually appear in that group's grouped text, so the UI can select
evidence honestly and label the factor as a model-attributed predefined group.
"""
import os, re, csv, math, itertools, random
import numpy as np, torch
import app.models.multitask_infer as mi
from app.core.preprocessing import clean_text

FACTORS = {
  "Unrealistic income / commission": {"income","earn","earnings","commission","weekly","monthly","month","salary","pay","bonus","reach","unlimited","guaranteed","rich","reward","profit","earning"},
  "Remote / data-entry": {"remote","home","data","entry","typist","telecommute","wfh"},
  "Urgency / pressure": {"urgent","immediate","immediately","now","today","asap","hurry","quick","fast","act","limited","instantly"},
  "Low / no requirements": {"experience","qualifications","qualification","diploma","degree","required","requirement","training","inexperienced","anyone","everyone"},
  "Targeting language": {"female","male","gender","age","young","only","housewife","student","students"},
  "Contact / recruitment": {"whatsapp","telegram","call","phone","resume","cv","kindly","contact","dm","message","wa"},
  "Fees / payment request": {"fee","fees","deposit","registration","upfront","refundable","charge","charges"},
  "Personal / bank information": {"bank","account","passport","identity","ssn","card","details","ic"},
}
OTHER = "Other posting content"
NUM_BACKGROUNDS = 10
CHUNK = 64
WINDOW = 25
MAXTOK = 256
MAXGROUPS = 7

# ── Sentence-scoped negation gate for factor grouping ────────────────────────
# The FACTORS keyword sets are polarity-blind: `registration`/`fee`/`charge`
# fire on "No registration fee required" the same as on "Applicants must pay a
# registration fee". Without a negation gate, the whole disclaimer sentence
# ends up in the Fees group, all 14 of its word positions get replaced when
# Shapley computes v(S \ {Fees}), and the intervention picks up a large,
# semantically misleading contribution.
#
# _is_neg_at asks: is the cleaned word at index i in this unit's word list
# preceded within _NEG_WINDOW words by a common English negator? Because cw
# is one _segment unit's cleaned words (units are sentence-scoped), this is
# automatically sentence-scoped: a negator in an earlier sentence cannot
# cancel a keyword in a later one.
#
# _NEGATION_INSENSITIVE_FACTORS mirrors the opt-in idiom used in patterns.py
# and in the frontend evidence gate. "Low / no requirements" matches on
# negation BY DESIGN — "no experience", "no diploma" ARE the scam signal
# there — so it stays negation-insensitive. Every other factor is polarity-
# aware.
#
# clean_text() strips apostrophes, so contractions ending in n't never reach
# this list; only the plain-word negators below are effective here.
_NEG_WINDOW = 6
_SIMPLE_NEGATORS = frozenset({
    "no", "not", "never", "without", "cannot", "none", "neither", "nor",
})
_NEGATION_INSENSITIVE_FACTORS = frozenset({
    "Low / no requirements",
})

# ── Low / no requirements: specific-anchor rule ──────────────────────────────
# The Low keyword set contains SPECIFIC scam-signal words (experience,
# qualifications, diploma, degree, training, inexperienced, anyone, everyone)
# and AMBIGUOUS modifier words (required, requirement). "required" alone is
# ubiquitous in legitimate postings ("5 years of experience required",
# "signature required") and does not carry the low-requirements scam meaning
# on its own — it only carries that meaning when it appears with a concrete
# low-requirement anchor. Without this rule the sentence "no registration fee
# required" (whose Fees hits get zeroed by the negation gate) would still land
# in Low via the standalone "required", and the ~14 pts of Shapley mass that
# used to be a misleading Fees attribution would just be renamed to Low. With
# this rule, "required"/"requirement" count for Low only when the same unit
# also contains one of the specific Low keywords; otherwise they do not
# contribute to Low's tally and the sentence is free to fall through to OTHER.
_LOW_SPECIFIC_KEYWORDS = frozenset({
    "experience", "qualifications", "qualification", "diploma", "degree",
    "inexperienced", "anyone", "everyone", "training",
})
_LOW_AMBIGUOUS_KEYWORDS = frozenset({"required", "requirement"})
_LOW_LABEL = "Low / no requirements"

_BGS = None

def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def _find_val():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.path.join(here, "..", "..", "..", "ml", "data", "processed", "val.csv"),
              os.path.join(os.getcwd(), "..", "ml", "data", "processed", "val.csv"),
              os.path.join(os.getcwd(), "ml", "data", "processed", "val.csv")]:
        if os.path.isfile(c):
            return os.path.normpath(c)
    return None

def _load_backgrounds():
    global _BGS
    if _BGS is not None:
        return _BGS
    path = _find_val()
    if not path:
        raise RuntimeError("faithful_xai: val.csv background file not found")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    gen = [r["clean_text"] for r in rows
           if str(r["fraudulent"]).strip() in ("0", "0.0") and len(r["clean_text"].split()) > 40]
    random.seed(0)
    _BGS = [t.split() for t in random.sample(gen, min(NUM_BACKGROUNDS, len(gen)))]
    return _BGS

def _score_batch(texts):
    mi._load()
    out = []
    for i in range(0, len(texts), CHUNK):
        enc = mi._mt_tok(texts[i:i+CHUNK], truncation=True, max_length=MAXTOK,
                         padding=True, return_tensors="pt").to(mi._mt_device)
        with torch.inference_mode():
            logits, _ = mi._mt_model(enc["input_ids"], enc["attention_mask"])
        out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

def _split_sentences(raw):
    return [p.strip() for p in re.split(r'(?<=[.!?])\s+|\n+|\u2022|;', raw) if p.strip()]

def _segment(raw):
    units = []
    for s in _split_sentences(raw):
        cw = clean_text(s).split()
        if not cw:
            continue
        if len(cw) <= WINDOW:
            units.append({"raw": s.strip(), "words": cw})
        else:
            for i in range(0, len(cw), WINDOW):
                units.append({"raw": " ".join(cw[i:i+WINDOW]), "words": cw[i:i+WINDOW]})
    return units

def _is_neg_at(cw, i):
    """True when the cleaned word at index i is preceded within
    _NEG_WINDOW words by a simple negator inside the same unit."""
    start = i - _NEG_WINDOW if i > _NEG_WINDOW else 0
    for j in range(start, i):
        if cw[j] in _SIMPLE_NEGATORS:
            return True
    return False

def _unit_group(cw):
    best, n = OTHER, 0
    # Precompute: does the unit contain a specific Low anchor keyword?
    # Used to decide whether the ambiguous Low keywords ("required",
    # "requirement") contribute to Low's tally.
    low_has_specific_anchor = any(w in _LOW_SPECIFIC_KEYWORDS for w in cw)
    for name, kws in FACTORS.items():
        neg_sensitive = name not in _NEGATION_INSENSITIVE_FACTORS
        is_low = name == _LOW_LABEL
        c = 0
        for i, w in enumerate(cw):
            if w not in kws:
                continue
            if neg_sensitive and _is_neg_at(cw, i):
                continue
            # Specific-anchor rule for Low: an ambiguous modifier word
            # only counts when a specific Low anchor keyword is also
            # present in the same unit.
            if is_low and w in _LOW_AMBIGUOUS_KEYWORDS and not low_has_specific_anchor:
                continue
            c += 1
        if c > n:
            n, best = c, name
    return best

def _build(raw):
    wg, units = [], []
    for u in _segment(raw):
        cw = u["words"]; g = _unit_group(cw); widx = list(range(len(wg), len(wg)+len(cw)))
        for w in cw:
            wg.append((w, g))
        units.append({"raw": u["raw"], "words": cw, "group": g, "widx": widx})
    counts = {}
    for _, g in wg:
        counts[g] = counts.get(g, 0) + 1
    present = [g for g in list(FACTORS.keys()) + [OTHER] if counts.get(g, 0) > 0]
    if len(present) > MAXGROUPS:
        named = sorted([g for g in present if g != OTHER], key=lambda g: counts[g])
        while len(present) > MAXGROUPS and named:
            d = named.pop(0)
            wg = [(w, OTHER if g == d else g) for w, g in wg]
            for u in units:
                if u["group"] == d:
                    u["group"] = OTHER
            present.remove(d)
        if OTHER not in present:
            present.append(OTHER)
    return units, wg, present, counts

def _mixed(wg, S, bg):
    n = len(bg)
    return " ".join(w if g in S else bg[i % n] for i, (w, g) in enumerate(wg))

def _tok_len(t):
    mi._load()
    return len(mi._mt_tok(t, truncation=False, add_special_tokens=False)["input_ids"])

def explain(raw_text):
    bgs = _load_backgrounds()
    units, wg, present, counts = _build(raw_text)
    recon_text = " ".join(w for w, _ in wg)
    if not present or not recon_text.strip():
        return {"available": False, "reason": "No analysable content.",
                "final_pct": round(float(_score_batch([clean_text(raw_text)])[0]) * 100, 2),
                "reference_pct": None, "reconstruct_pct": None, "truncated": None,
                "flagged": [], "stylistic": None, "weak_explanation": False}

    f_recon = float(_score_batch([recon_text])[0])

    subsets = [frozenset(c) for r in range(len(present) + 1)
               for c in itertools.combinations(present, r)]
    pos, texts = {}, []
    for S in subsets:
        for bi, bg in enumerate(bgs):
            pos[(S, bi)] = len(texts)
            texts.append(_mixed(wg, S, bg))
    scores = _score_batch(texts)
    vcache = {S: float(np.mean([scores[pos[(S, bi)]] for bi in range(len(bgs))])) for S in subsets}

    k = len(present)
    phi = {g: 0.0 for g in present}
    for g in present:
        others = [x for x in present if x != g]
        for r in range(len(others) + 1):
            wgt = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
            for combo in itertools.combinations(others, r):
                S = frozenset(combo)
                phi[g] += wgt * (vcache[S | {g}] - vcache[S])
    reference = vcache[frozenset()]

    cum, cut = 0, None
    for i, u in enumerate(units):
        cum += _tok_len(" ".join(u["words"]))
        if cut is None and cum > MAXTOK:
            cut = i
    seen = [i for i in range(len(units)) if cut is None or i < cut]
    bg0 = bgs[0]; n0 = len(bg0)
    ev_texts = []
    for i in seen:
        idx = set(units[i]["widx"])
        ev_texts.append(" ".join((bg0[j % n0] if j in idx else w) for j, (w, _) in enumerate(wg)))
    ev_scores = _score_batch(ev_texts) if ev_texts else np.zeros(0)
    eff = {seen[j]: (f_recon - float(ev_scores[j])) for j in range(len(seen))}

    flagged = []
    for g in sorted(present, key=lambda g: -abs(phi[g])):
        us = sorted([i for i in seen if units[i]["group"] == g], key=lambda i: -abs(eff.get(i, 0.0)))[:3]
        # Display-only: surface the top model-tagged unit texts for this factor.
        # Previously filtered by |eff| > 0.002, which emptied `words` for factors
        # whose Shapley mass was distributed across units (e.g. Remote, Targeting),
        # leaving the frontend evidence fallback with nothing to show. The filter is
        # dropped; `words` never enters the Shapley score, so reference + Sum(phi) =
        # final is unchanged.
        words = [units[i]["raw"][:80] for i in us]
        # category-defining keywords actually present in this factor's grouped text
        grouped = set(w for (w, gg) in wg if gg == g)
        keywords = sorted(grouped & FACTORS.get(g, set()))
        flagged.append({"id": _slug(g), "title": g, "points": round(phi[g] * 100, 2),
                        "words": words, "keywords": keywords})

    return {
        "available": True, "reason": None,
        "final_pct": round(f_recon * 100, 2),
        "reference_pct": round(reference * 100, 2),
        "reconstruct_pct": round((reference + sum(phi.values())) * 100, 2),
        "flagged": flagged, "stylistic": None, "weak_explanation": False,
        "truncated": cut is not None,
    }