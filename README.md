# 🔬 Clinical AI Failure Observatory

[![CI](https://github.com/enghamza-AI/Orion/actions/workflows/ci.yml/badge.svg)](https://github.com/enghamza-AI/Orion/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/enghamza-AI/Orion)


> **Upload any tabular medical dataset. Get the truth about your model in 90 seconds.**

Clinical AI systems fail silently. A model can report 94% accuracy in development and collapse to 61% AUC in production — not because the model is wrong, but because the evaluation was. Leaky pipelines, corrupted data, and structural overfitting go undetected until patients are affected.

The Observatory is a one-click diagnostic tool that runs four independent engines against any dataset and produces a **Model Trust Score** with a downloadable PDF audit report.

---

## 🎯 The Problem This Solves

| What teams do | What actually happens |
|---|---|
| Report accuracy on imbalanced data | 91% accuracy = model always predicts majority class |
| Fit StandardScaler on all data | Test set statistics leak into training |
| Leave `lab_result_final` as a feature | Column derived from target — model is cheating |
| Train one model, ship it | Never checked if it overfits on this specific dataset |
| Do EDA and call the data "clean" | EDA finds formatting issues. Not ML-specific failure modes. |

---

## 🔬 The Four Diagnostic Engines

### Engine 1 — Noise Auditor
Inspects raw data for 6 corruption archetypes **before any model is involved**:
- Missing values (sensor dropout)
- Statistical outliers (impossible values)
- Exact duplicate rows
- Class imbalance
- Near-constant (zero-information) columns
- Data type mismatches

### Engine 2 — Bias-Variance Engine
Trains 5 models of increasing complexity and diagnoses structural failure mode:
- `UNDERFIT` — model too simple for this data (high bias)
- `OVERFIT` — model memorizing training data (high variance)
- `BALANCED` — train and test AUC converged (healthy)

### Engine 3 — Leakage Scanner
Detects 5 data leakage archetypes that cause inflated development metrics:
- **Sin 1:** Target encoding computed before train/test split
- **Sin 2:** Feature with suspiciously high solo predictive power (>0.90 AUC alone)
- **Sin 3:** StandardScaler fit on full dataset including test set
- **Sin 4:** Same entity (patient/borrower) in both train and test
- **Sin 5:** Exact duplicate rows across train/test boundary

Each detected sin reports measured AUC inflation — the exact number of fake performance points it generated.

### Engine 4 — Learning Curve Autopsy
Trains on 10% → 20% → ... → 100% of data. Diagnoses:
- `DATA-STARVED` — adding more data won't help; fix the features
- `OVER-COMPLEX` — model always memorizes regardless of data size
- `LEAKY` — validation AUC suspiciously exceeds training AUC
- `HEALTHY` — curves converging at high values

---

## 💎 Model Trust Score

All engine findings feed a **penalty-based scoring system**:
Score = 100
− up to 25 pts  (Noise Audit)
− up to 25 pts  (Bias-Variance)
− up to 30 pts  (Leakage — highest weight: leakage = fake results)
− up to 20 pts  (Curve Autopsy)

| Score | Grade | Verdict |
|-------|-------|---------|
| 90–100 | A | Trustworthy — deploy with monitoring |
| 75–89  | B | Mostly trustworthy — fix flagged items |
| 60–74  | C | Caution — not for high-stakes decisions |
| 40–59  | D | Significant issues — rework required |
| 0–39   | F | Do not deploy — results cannot be trusted |

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/enghamza-AI/Orion.git
cd orion

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**No dataset?** Click "Use Demo Dataset" — a synthetic Home Credit-style dataset
with injected corruption patterns loads automatically.

---

## 📁 Project Structure
Orion/
│
├── app.py                          # Streamlit UI — single entry point
│
├── engines/
│   ├── noise_auditor.py            # Engine 1: 6 corruption checks
│   ├── bias_variance_engine.py     # Engine 2: 5-model complexity ladder
│   ├── leakage_scanner.py          # Engine 3: 5 leakage sin detectors
│   └── curve_autopsy.py            # Engine 4: learning curve + diagnosis
│
├── core/
│   ├── orchestrator.py             # Runs all 4 engines, combines output
│   ├── trust_score.py              # Penalty-based 0–100 scoring system
│   └── pdf_reporter.py             # 6-page PDF report generator
│
├── data/
│   └── demo_sample.csv             # Demo dataset (synthetic, injected issues)
│
├── tests/
│   └── test_engines.py             # 20 pytest tests across all engines
│
└── .github/workflows/ci.yml        # GitHub Actions CI pipeline

---

## 🧪 Running Tests

```bash
# Run full test suite
pytest tests/test_engines.py -v

# Run with coverage report
pytest tests/test_engines.py --cov=engines --cov=core --cov-report=term-missing
```

---

## 🏗️ Technical Architecture
CSV Upload
│
▼
Orchestrator.run_all()
├── NoiseAuditor.audit()          → noise findings dict
├── BiasVarianceEngine.run()      → model results dict
├── LeakageScanner.scan()         → leakage findings dict
└── CurveAutopsy.run()            → curve data + diagnosis
│
▼
TrustScoreEngine.compute()           → score, grade, verdict, breakdown
│
▼
PDFReporter.generate()               → PDF bytes for download
│
▼
Streamlit app.py                     → renders everything in browser

---

## 🎯 Why This Matters

This is what clinical AI validation tools look like internally at health AI
companies. When a model fails in production, it's almost never because the
algorithm was wrong. It's because of exactly the failure modes this tool detects:

- A feature derived from the target (leakage) inflating AUC by 0.18
- A StandardScaler fit on all data (leakage) inflating AUC by 0.04
- Class imbalance making accuracy meaningless
- Overfitting that only shows up at deployment scale

The Observatory makes these invisible problems visible — automatically,
consistently, on any dataset.

---

## 📊 Demo

**Live on HuggingFace Spaces:**
👉 [huggingface.co/spaces/enghamza-AI/Orion](https://huggingface.co/spaces/enghamza-AI/Orion)

---

## 🛠️ Tech Stack

`Python 3.10+` · `Streamlit` · `scikit-learn` · `pandas` · `numpy` ·
`matplotlib` · `ReportLab` · `GitHub Actions` · `HuggingFace Spaces`

---

## 👤 Author

**enghamza-AI**
BSAI Student · Self-studying AI Systems Engineering
- GitHub: [github.com/enghamza-AI](https://github.com/enghamza-AI)
- HuggingFace: [huggingface.co/spaces/enghamza-AI](https://huggingface.co/spaces/enghamza-AI)

---

