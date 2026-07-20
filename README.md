# Explainable AI Fraudulent Job Posting Detector

Final Year Project — B.Sc. (Hons) Computer Science, Specialism in Artificial Intelligence
Asia Pacific University of Technology and Innovation

**Student:** Sehba Hanief (TP078886)
**Supervisor:** Assoc. Prof. Dr. Muhammad Ehsan Rana

## Project Status

🚧 **Phase 1 — Project Foundation & Environment Setup** (current phase)

This repository currently contains only the project skeleton. No functional
code has been implemented yet. See [Development Phases](#development-phases)
below for the full roadmap.

## Overview

This project detects fraudulent online job postings using Natural Language
Processing, deep learning, and Optical Character Recognition (OCR), with
an Explainable AI (XAI) layer that highlights the reasoning behind each
prediction. It is based on the Investigation Report submitted for this FYP,
which used the Employment Scam Aegean Dataset (EMSCAD) as the primary
data source.

## Project Structure

```
fraud-detector-fyp/
├── frontend/     React web application (user interface)
├── backend/      FastAPI server (API, OCR, XAI, database access)
├── ml/           Model training, experimentation, and evaluation
└── docs/         Architecture notes and supporting documentation
```

See `docs/` for a more detailed breakdown of each folder's purpose.

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React (Vite) |
| Backend | FastAPI |
| ML / Deep Learning | PyTorch, Hugging Face Transformers (BERT) |
| Baseline Models | scikit-learn |
| Explainable AI | LIME, SHAP |
| OCR | Tesseract / EasyOCR |
| Database | SQLite |

## Development Phases

- [x] **Phase 1** — Project foundation & environment setup *(in progress)*
- [ ] **Phase 2** — Data collection & preprocessing (EMSCAD)
- [ ] **Phase 3** — Model development (baselines + BERT fine-tuning) & Explainable AI integration
- [ ] **Phase 4** — OCR integration
- [ ] **Phase 5** — Backend API development
- [ ] **Phase 6** — Frontend development
- [ ] **Phase 7** — Evaluation & reporting
- [ ] **Phase 8** — Polish, testing & demo preparation

## Getting Started

> Setup instructions (virtual environment activation, dependency installation,
> running the frontend/backend) will be added here once Phase 1 environment
> setup is complete.

## License

Academic project — Asia Pacific University of Technology and Innovation.
