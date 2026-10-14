import os, json
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel

_MT_DIR = os.path.normpath(os.path.join("..", "ml", "saved_models", "bert_fraud_detector_multitask_v3"))
_mt_model = None
_mt_tok = None
_mt_tags = None
_mt_device = None

class MultiTaskBert(nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h, 2)
        self.token_head = nn.Linear(h, n_tags)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        pooled = seq[:, 0]
        logits = self.classifier(self.dropout(pooled))
        token_logits = self.token_head(seq)
        return logits, token_logits

def _load():
    global _mt_model, _mt_tok, _mt_tags, _mt_device
    if _mt_model is not None:
        return
    _mt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(_MT_DIR, "tags.json")) as f:
        _mt_tags = json.load(f)["tags"]
    _mt_tok = BertTokenizerFast.from_pretrained(_MT_DIR)
    _mt_model = MultiTaskBert(len(_mt_tags))
    state = torch.load(os.path.join(_MT_DIR, "multitask_model.pt"), map_location=_mt_device)
    _mt_model.load_state_dict(state)
    _mt_model.to(_mt_device)
    _mt_model.eval()

TAG_INFO = {
    "FEE": ("money_requests", "Requests for money or financial details"),
    "CONTACT": ("external_contact", "Contact through unofficial channels"),
    "INCOME": ("unrealistic_income", "Unrealistic income promises"),
    "IDENTITY": ("identity_requests", "Personal identity information requests"),
    "URGENCY": ("pressure", "Pressure and urgency tactics"),
    "LOWREQ": ("low_requirements", "Suspiciously low entry requirements"),
    "TARGETING": ("targeting", "Unprofessional or discriminatory targeting"),
    "REMOTE_VAGUE": ("remote_vague", "Vague remote or data-entry role"),
    "PURCHASE_EQUIP": ("purchase_equipment", "Required to purchase equipment or kit"),
    "CRYPTO_MULE": ("crypto_mule", "Cryptocurrency or money-transfer tasks"),
    "RESHIPPING": ("reshipping", "Receiving or forwarding money or packages"),
}

_DISPLAY_STOPWORDS = {
    "at","us","to","on","the","a","an","of","in","for","and","or","your",
    "our","we","you","is","are","be","with","via","by","this","that","it",
    "as","from","me","my","please","kindly","will","can","have","has",
}


def multitask_fraud_prob(text, clean_fn=None):
    _load()
    t = clean_fn(text) if clean_fn else text
    enc = _mt_tok([t], truncation=True, max_length=256, padding=True, return_tensors="pt").to(_mt_device)
    with torch.no_grad():
        logits, _ = _mt_model(enc["input_ids"], enc["attention_mask"])
    return torch.softmax(logits, dim=1)[0, 1].item()

def multitask_breakdown(text, clean_fn, fraud_prob):
    _load()
    final_pct = round(fraud_prob * 100, 1)
    t = clean_fn(text) if clean_fn else text
    enc = _mt_tok([t], truncation=True, max_length=256, padding=True, return_tensors="pt")
    enc = {k: v.to(_mt_device) for k, v in enc.items()}
    with torch.no_grad():
        _, token_logits = _mt_model(enc["input_ids"], enc["attention_mask"])
    probs = torch.softmax(token_logits, dim=2)[0].cpu().numpy()
    tag_idx = {tag: i for i, tag in enumerate(_mt_tags)}
    input_ids = enc["input_ids"][0].cpu().tolist()
    toks = _mt_tok.convert_ids_to_tokens(input_ids)
    words_merged = []
    cur_word = None
    cur_probs = None
    for i, tok in enumerate(toks):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if tok.startswith("##") and cur_word is not None:
            cur_word += tok[2:]
            cur_probs.append(probs[i])
        else:
            if cur_word is not None:
                words_merged.append((cur_word, cur_probs))
            cur_word = tok
            cur_probs = [probs[i]]
    if cur_word is not None:
        words_merged.append((cur_word, cur_probs))
    cat_score = {}
    cat_words = {}
    for word, plist in words_merged:
        avg = np.mean(plist, axis=0)
        best_sig = None
        best_p = 0.0
        for tag, ti in tag_idx.items():
            if tag == "O":
                continue
            sig = tag.split("-")[-1]
            p = avg[ti]
            if p > best_p:
                best_p = p
                best_sig = sig
        if best_sig and best_p > 0.15:
            cat_score[best_sig] = cat_score.get(best_sig, 0.0) + float(best_p)
            wl = word.lower().strip()
            if wl and wl not in _DISPLAY_STOPWORDS and len(wl) > 1:
                cat_words.setdefault(best_sig, []).append(word)
    if sum(cat_score.values()) <= 0:
        return {"available": False, "reason": "The model did not attribute the score to specific signals.", "final_pct": final_pct, "flagged": [], "stylistic": None}
    n_sig = len(cat_score)
    if n_sig >= 2:
        cat_score = {k: (v ** 0.5) for k, v in cat_score.items()}
        MAX_SHARE = 0.55 if n_sig >= 3 else 0.65
        for _ in range(4):
            tot = sum(cat_score.values())
            over = {k: v for k, v in cat_score.items() if v / tot > MAX_SHARE}
            if not over:
                break
            for k in over:
                cap_val = MAX_SHARE * tot
                excess = cat_score[k] - cap_val
                cat_score[k] = cap_val
                others = {j: cat_score[j] for j in cat_score if j != k}
                osum = sum(others.values()) or 1.0
                for j in others:
                    cat_score[j] += excess * (others[j] / osum)
    total = sum(cat_score.values())
    flagged = []
    for sig, sc in cat_score.items():
        cid, title = TAG_INFO.get(sig, (sig.lower(), sig.title()))
        words = []
        for w in cat_words.get(sig, []):
            if not words or words[-1] != w:
                words.append(w)
        flagged.append({"id": cid, "title": title, "points": round(final_pct * (sc / total), 1), "words": words[:6]})
    flagged.sort(key=lambda f: -f["points"])
    diff = round(final_pct - sum(f["points"] for f in flagged), 1)
    if flagged and abs(diff) >= 0.1:
        flagged[0]["points"] = round(flagged[0]["points"] + diff, 1)
    strong_signals = [f for f in flagged if f["points"] >= 8]
    weak_explanation = (fraud_prob >= 0.5) and (len(strong_signals) < 2)
    return {"available": True, "reason": None, "final_pct": final_pct, "flagged": flagged, "stylistic": None, "weak_explanation": weak_explanation}
