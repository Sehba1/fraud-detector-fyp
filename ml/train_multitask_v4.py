import os, json, numpy as np, torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
EPOCHS = 3
BATCH = 16
LR = 2e-5
TOKEN_LOSS_WEIGHT = 0.5
OUTPUT_DIR = "bert_fraud_detector_multitask_v4"

TAGS = ["O","B-FEE","I-FEE","B-CONTACT","I-CONTACT","B-INCOME","I-INCOME",
        "B-IDENTITY","I-IDENTITY","B-URGENCY","I-URGENCY","B-LOWREQ","I-LOWREQ",
        "B-TARGETING","I-TARGETING","B-REMOTE_VAGUE","I-REMOTE_VAGUE",
        "B-PURCHASE_EQUIP","I-PURCHASE_EQUIP","B-CRYPTO_MULE","I-CRYPTO_MULE",
        "B-RESHIPPING","I-RESHIPPING"]
TAG2ID = {t:i for i,t in enumerate(TAGS)}
ID2TAG = {i:t for t,i in TAG2ID.items()}

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def load_jsonl(path):
    rows=[]
    for line in open(path, encoding="utf-8"):
        line=line.strip()
        if line: rows.append(json.loads(line))
    return rows

def load_csv_classification(path):
    import pandas as pd
    df = pd.read_csv(path)
    # find the text column
    text_col = "clean_text" if "clean_text" in df.columns else ("text" if "text" in df.columns else None)
    lab_col = "fraudulent" if "fraudulent" in df.columns else ("label" if "label" in df.columns else None)
    rows = []
    for _, d in df.iterrows():
        text = str(d[text_col]) if text_col and pd.notna(d[text_col]) else ""
        if not text.strip():
            continue
        try:
            lab = int(float(d[lab_col]))
        except (ValueError, TypeError):
            lab = 0
        rows.append({"text": text, "label": lab, "tokens": None, "tags": None})
    print(f"  loaded {len(rows)} rows from {path}")
    return rows

class MTDataset(Dataset):
    def __init__(self, rows):
        self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]
        if r.get("tokens"):
            enc=tokenizer(r["tokens"], is_split_into_words=True,
                          truncation=True, max_length=MAX_LEN,
                          padding="max_length", return_tensors="pt")
            word_ids=enc.word_ids(0)
            labels=[]
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)
                else:
                    tag=r["tags"][wid] if wid < len(r["tags"]) else "O"
                    labels.append(TAG2ID.get(tag, 0))
            has_token=1
        else:
            enc=tokenizer(r["text"], truncation=True, max_length=MAX_LEN,
                          padding="max_length", return_tensors="pt")
            labels=[-100]*MAX_LEN
            has_token=0
        return {
            "input_ids":enc["input_ids"][0],
            "attention_mask":enc["attention_mask"][0],
            "class_label":torch.tensor(r["label"],dtype=torch.long),
            "token_labels":torch.tensor(labels,dtype=torch.long),
            "has_token":torch.tensor(has_token,dtype=torch.float),
        }

class MultiTaskBert(nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.bert=BertModel.from_pretrained(MODEL_NAME)
        h=self.bert.config.hidden_size
        self.dropout=nn.Dropout(0.1)
        self.classifier=nn.Linear(h,2)
        self.token_head=nn.Linear(h,n_tags)
    def forward(self, input_ids, attention_mask):
        out=self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq=out.last_hidden_state
        pooled=seq[:,0]
        logits=self.classifier(self.dropout(pooled))
        token_logits=self.token_head(seq)
        return logits, token_logits

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    rows=load_jsonl("token_labeled_train_v4.jsonl")
    print("Token-labeled rows:", len(rows))
    if os.path.exists("train.csv"):
        emscad=load_csv_classification("train.csv")
        print("EMSCAD classification rows:", len(emscad))
        rows = rows + emscad

    ds=MTDataset(rows)
    dl=DataLoader(ds, batch_size=BATCH, shuffle=True)

    model=MultiTaskBert(len(TAGS)).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps=len(dl)*EPOCHS
    sched=get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

    class_loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1)
    token_loss_fn=nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for ep in range(EPOCHS):
        running=0.0
        for step,batch in enumerate(dl):
            ids=batch["input_ids"].to(device)
            mask=batch["attention_mask"].to(device)
            cl=batch["class_label"].to(device)
            tl=batch["token_labels"].to(device)
            ht=batch["has_token"].to(device)

            logits, token_logits=model(ids, mask)
            loss_c=class_loss_fn(logits, cl)
            if ht.sum() > 0:
                tlogits=token_logits.view(-1, len(TAGS))
                tlabels=tl.view(-1)
                loss_t=token_loss_fn(tlogits, tlabels)
            else:
                loss_t=torch.tensor(0.0, device=device)
            loss=loss_c + TOKEN_LOSS_WEIGHT*loss_t
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step()
            running+=loss.item()
            if step % 100 == 0:
                print(f"  epoch {ep+1} step {step}/{len(dl)} loss {loss.item():.4f}")
        print(f"Epoch {ep+1} avg loss {running/len(dl):.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR,"multitask_model.pt"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR,"tags.json"),"w") as f:
        json.dump({"tags":TAGS}, f)
    print("Saved to", OUTPUT_DIR)

    if os.path.exists("val.csv"):
        val=load_csv_classification("val.csv")
        model.eval()
        preds=[]; labs=[]; probs=[]
        with torch.no_grad():
            for i in range(0,len(val),BATCH):
                chunk=val[i:i+BATCH]
                enc=tokenizer([c["text"] for c in chunk], truncation=True,
                              max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
                logits,_=model(enc["input_ids"], enc["attention_mask"])
                p=torch.softmax(logits,1)[:,1].cpu().numpy()
                preds.extend((p>0.5).astype(int)); probs.extend(p)
                labs.extend([c["label"] for c in chunk])
        print(classification_report(labs, preds, target_names=["genuine","fraudulent"], digits=4))
        print("ROC-AUC:", round(roc_auc_score(labs, probs),4))

if __name__ == "__main__":
    main()