# FraudLens: Explainable AI Job Scam Detector

FraudLens detects fraudulent online job postings and explains **why** each posting was flagged. A fine-tuned BERT model produces a fraud probability, a Group-Shapley explanation layer breaks that score down into human-readable scam-risk factors, and a separate rule-based layer flags known scam patterns. It runs as a React web app and a Chrome extension backed by a FastAPI service.

Final Year Project, B.Sc. (Hons) Computer Science (Artificial Intelligence), Asia Pacific University of Technology & Innovation (APU), 2026.

## Results

Final model (multitask V3) evaluated on the held-out EMSCAD validation set of **2,682 postings** (130 fraudulent):

| Metric | Value |
|---|---|
| Accuracy | 0.986 |
| Fraud precision | 0.919 |
| Fraud recall | 0.785 |
| Fraud F1-score | 0.847 |
| ROC-AUC | 0.984 |

## Features

- **BERT classifier:** fine-tuned `bert-base-uncased` multitask model (V3) that outputs a fraud probability, a genuine/fraudulent label and a low/medium/high risk tier.
- **Faithful explanations:** an interventional Group-Shapley method attributes the fraud probability to 8 domain-informed factor groups (unrealistic income, remote/data-entry framing, urgency, low requirements, targeting language, unofficial contact, fee requests, personal/bank information requests). The baseline plus the factor contributions reconstruct the final score exactly.
- **Real evidence:** each factor is backed by a sentence quoted from the original posting, or reported as "no specific evidence identified". Evidence is never generated.
- **Rule-based checks:** a deterministic, negation-aware and context-aware pattern layer. For example, "no registration fee required" is not flagged as a fee request. This layer is kept separate from the model's explanation.
- **Text and image input:** paste a posting, or upload a screenshot of a job ad that is read with OCR.
- **Chrome extension (Manifest V3):** analyse a posting directly on a recruitment site. If the job description can't be identified confidently, it asks you to select or paste it instead of sending the whole page.
- **History:** submissions, predictions and explanations are stored in SQLite.

## Architecture

```
React web app (Vite) ─┐
                      ├──> FastAPI backend ──> V3 BERT model ──> fraud probability
Chrome extension ─────┘          │            Group-Shapley  ──> factor breakdown + evidence
                                 │            Rule patterns  ──> warning / genuine indicators
                                 └──> SQLite (submissions, predictions, explanations)
```

## Repository structure

```
backend/     FastAPI app (app/api, app/core, app/models, app/schemas) and requirements.txt
frontend/    React + Vite web app
extension/   Chrome extension (Manifest V3)
ml/          Training scripts and notebooks (final model: train_multitask_v3.py)
```

## Model weights and data (not included)

The trained models and datasets are **not in this repository**, because the BERT checkpoints are several hundred MB each. The backend checks for these files at startup, and the paths are set in `backend/app/core/config.py`:

| File | Expected location |
|---|---|
| Final BERT model (multitask V3) | `ml/saved_models/bert_fraud_detector_multitask_v3/` |
| Baseline Random Forest | `ml/saved_models/baseline_random_forest.joblib` |
| TF-IDF vectorizer | `ml/saved_models/tfidf_vectorizer.joblib` |

**Dataset:** [Employment Scam Aegean Dataset (EMSCAD)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), available on Kaggle. Download it, create the train/validation split, and place the files in `ml/data/`. The exact file paths are at the top of `ml/train_multitask_v3.py`.

**To retrain the final model:**

```bash
python ml/train_multitask_v3.py
```

A GPU is strongly recommended. The notebooks in `ml/notebooks/` cover EDA, baseline models, BERT fine-tuning and evaluation.

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Google Chrome (or another Chromium-based browser) for the extension
- The model files above in place

### 1. Backend (FastAPI)

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # macOS/Linux: cp .env.example .env  (no secrets are required)
uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

### 2. Frontend (React + Vite)

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

### 3. Chrome extension

1. Open `chrome://extensions`.
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and select the `extension/` folder.
4. With the backend running, open a job posting, click the FraudLens icon, and analyse the page or your selected text.

## Tech stack

Python, Hugging Face Transformers, BERT, FastAPI, SQLAlchemy, SQLite, OCR, React, Vite, JavaScript, Chrome Extensions (Manifest V3), Shapley-value explainability (Group-Shapley)

## Limitations

- English-only (`bert-base-uncased`, trained on English postings).
- Explanations take longer for postings that trigger many factor groups.
- The browser extension is a prototype and may ask you to select the job text on some sites.
- Performance reflects the training data, so postings very different from EMSCAD may be classified less reliably.

FraudLens gives a risk estimate, not a guarantee. Always verify a job offer independently.

## Author

**Sehba Hanief**
[LinkedIn](https://www.linkedin.com/in/sehba-hanief-a64818329/) · [GitHub](https://github.com/Sehba1)

Supervised by Assoc. Prof. Dr. Muhammad Ehsan Rana, Asia Pacific University of Technology & Innovation.

This project was built for academic and portfolio purposes.
