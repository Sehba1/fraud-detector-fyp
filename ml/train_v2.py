"""
Retrain the fraud detection model on EMSCAD combined with collected
augmented data.

Run this on Google Colab with a GPU runtime. It expects these files to be
uploaded to the Colab session:

    train.csv                 - the existing EMSCAD training split
    val.csv                   - the existing EMSCAD validation split
    augmented_train.jsonl     - output of ml/generate_variations.py
    holdout_real.jsonl        - held-out real examples, optional

Design notes for the project report:

  - The augmented set is small relative to EMSCAD (roughly 450 examples
    against 17,880). Without weighting, its influence on the fitted model
    would be negligible. AUGMENT_REPEAT therefore oversamples the
    augmented examples so that the scam classes they represent carry
    weight proportionate to their importance rather than their raw count.
    This is standard practice for rare classes and is disclosed here.

  - The augmented examples are added to the TRAINING split only. The
    EMSCAD validation and test splits are left untouched, so the reported
    EMSCAD figures remain directly comparable with the original model.

  - The held-out real examples (holdout_real.jsonl) are never seen during
    training and are evaluated separately after training completes.

  - Training is from scratch on bert-base-uncased rather than continued
    from the previous fine-tuned checkpoint, so the result is a clean
    single-stage fine-tune that is straightforward to describe.
"""

import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512
EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SEED = 42

# How many times each augmented example is repeated in the training set.
# See the module docstring for the rationale.
AUGMENT_REPEAT = 8

OUTPUT_DIR = "bert_fraud_detector_v2"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Data loading ──────────────────────────────────────────────────────────────
def clean_text(text):
    """
    Minimal cleaning, matching the preprocessing used for the original
    model so the two are comparable.
    """
    import re
    from bs4 import BeautifulSoup

    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = re.sub(r"#[A-Z_]+#", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


print("Loading EMSCAD splits...")
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

text_col = "clean_text" if "clean_text" in train_df.columns else "text"
label_col = "fraudulent" if "fraudulent" in train_df.columns else "label"

train_texts = train_df[text_col].astype(str).tolist()
train_labels = train_df[label_col].astype(int).tolist()
val_texts = val_df[text_col].astype(str).tolist()
val_labels = val_df[label_col].astype(int).tolist()

print(f"  EMSCAD train: {len(train_texts)} "
      f"({sum(train_labels)} fraudulent)")
print(f"  EMSCAD val:   {len(val_texts)} "
      f"({sum(val_labels)} fraudulent)")

print("\nLoading augmented data...")
aug_rows = load_jsonl("augmented_train.jsonl")

aug_texts = []
aug_labels = []
for row in aug_rows:
    cleaned = clean_text(row["text"])
    if not cleaned:
        continue
    label = 1 if row["label"] == "fraudulent" else 0
    # Oversample so the augmented classes carry meaningful weight.
    for _ in range(AUGMENT_REPEAT):
        aug_texts.append(cleaned)
        aug_labels.append(label)

print(f"  Augmented examples:        {len(aug_rows)}")
print(f"  Repeat factor:             {AUGMENT_REPEAT}")
print(f"  Contributed to training:   {len(aug_texts)}")
print(f"    fraudulent:              {sum(aug_labels)}")
print(f"    genuine:                 {len(aug_labels) - sum(aug_labels)}")

# Augmented data joins the training split only. Validation stays pure
# EMSCAD so the figures remain comparable with the original model.
combined_texts = train_texts + aug_texts
combined_labels = train_labels + aug_labels

print(f"\nCombined training set: {len(combined_texts)} "
      f"({sum(combined_labels)} fraudulent, "
      f"{100 * sum(combined_labels) / len(combined_labels):.2f}% positive)")


# ── Dataset ───────────────────────────────────────────────────────────────────
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


class PostingDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


print("\nTokenizing...")
train_dataset = PostingDataset(combined_texts, combined_labels)
val_dataset = PostingDataset(val_texts, val_labels)


# ── Weighted loss for class imbalance ─────────────────────────────────────────
n_pos = sum(combined_labels)
n_neg = len(combined_labels) - n_pos
pos_weight = n_neg / n_pos
print(f"\nClass weight on fraudulent: {pos_weight:.2f}")

class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float)


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Training ──────────────────────────────────────────────────────────────────
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=32,
    learning_rate=LEARNING_RATE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    seed=SEED,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\nTraining...")
trainer.train()

print(f"\nSaving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metadata = {
    "base_model": MODEL_NAME,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "max_length": MAX_LENGTH,
    "seed": SEED,
    "emscad_train_size": len(train_texts),
    "augmented_source_count": len(aug_rows),
    "augment_repeat": AUGMENT_REPEAT,
    "combined_train_size": len(combined_texts),
    "class_weight_fraudulent": float(pos_weight),
}
with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)


# ── Evaluation on the EMSCAD validation split ─────────────────────────────────
print("\n" + "=" * 60)
print("EMSCAD VALIDATION SET")
print("=" * 60)

preds = trainer.predict(val_dataset)
probs = torch.softmax(torch.tensor(preds.predictions), dim=1).numpy()
y_pred = (probs[:, 1] >= 0.5).astype(int)
y_true = np.array(val_labels)

print(classification_report(
    y_true, y_pred, target_names=["genuine", "fraudulent"], digits=4
))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_true, probs[:, 1]):.4f}")


# ── Evaluation on the held-out real examples ──────────────────────────────────
if os.path.exists("holdout_real.jsonl"):
    print("\n" + "=" * 60)
    print("HELD-OUT REAL EXAMPLES (never seen in training)")
    print("=" * 60)

    holdout = load_jsonl("holdout_real.jsonl")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0
    for row in holdout:
        cleaned = clean_text(row["text"])
        enc = tokenizer(
            [cleaned],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        p = torch.softmax(logits, dim=1)[0, 1].item()
        predicted = "fraudulent" if p >= 0.5 else "genuine"
        ok = predicted == row["label"]
        correct += ok
        mark = "correct" if ok else "WRONG  "
        print(f"  {mark}  {row['id']:<12} "
              f"actual={row['label']:<11} "
              f"predicted={predicted:<11} "
              f"fraud_prob={p:.4f}")

    print(f"\nHoldout accuracy: {correct}/{len(holdout)}")

print("\nDone. Download the folder:", OUTPUT_DIR)