# ICU Mortality & Length-of-Stay Analysis
### MIMIC-III · PhysioNet · Python · SQL · Power BI

> Identifying which ICU patients are at highest risk of in-hospital mortality and prolonged stays — and translating findings into actionable triage recommendations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![SQL](https://img.shields.io/badge/SQL-BigQuery-orange?style=flat-square)](https://cloud.google.com/bigquery)
[![Dashboard](https://img.shields.io/badge/Dashboard-Power%20BI-yellow?style=flat-square)](#dashboard)
[![Data](https://img.shields.io/badge/Data-MIMIC--III%20PhysioNet-green?style=flat-square)](https://physionet.org/content/mimiciii/1.4/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

---

## Contents

- [Project overview](#project-overview)
- [Key findings](#key-findings)
- [Dashboard](#dashboard)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [How to reproduce](#how-to-reproduce)
- [Methodology](#methodology)
- [Recommendations](#recommendations)
- [Data ethics & compliance](#data-ethics--compliance)
- [Contact](#contact)

---

## Project overview

Hospital ICU readmission and mortality are critical quality metrics tracked by healthcare systems worldwide. The US Centers for Medicare & Medicaid Services (CMS) penalise hospitals for avoidable adverse outcomes — making predictive analytics in this space a direct business problem, not just an academic exercise.

This project uses **58,976 adult ICU admissions** from the MIMIC-III clinical database to answer three business questions:

| # | Business question | Analytical approach |
|---|---|---|
| 1 | Which patient profiles carry the highest in-hospital mortality risk? | Cohort segmentation by age, admission type, care unit, and first-24h clinical values |
| 2 | What clinical signals on admission best predict mortality and prolonged stays (>7 days)? | Logistic regression + random forest with SHAP explainability |
| 3 | Do temporal patterns (weekend admissions, night-shift arrivals) affect outcomes? | Day-of-week and shift-level aggregation analysis |

---

## Key findings

**1. Age and care unit are the strongest structural risk factors**
Patients aged 76+ admitted to the MICU face a **26.2% in-hospital mortality rate** — more than 6× the rate of patients under 40 in surgical ICUs. Age is the single highest-importance feature in the predictive model (14.2% mean decrease in impurity).

**2. Lactate and creatinine are the most predictive lab values**
First-24h average lactate (11.8% importance) and creatinine (9.7%) outperform all other lab features. Elevated lactate is a well-established marker of tissue hypoperfusion — the model confirms its clinical significance quantitatively.

**3. The weekend effect is real and measurable**
Weekend admissions show a **+1.6 percentage point higher mortality rate** (12.6% vs 11.0%) and a **29% longer mean ICU LOS** for night-shift arrivals (4.9 days vs 3.8 days on day-shift). This is consistent with published literature on reduced specialist cover outside business hours.

**4. Emergency admissions carry 3.8× the mortality risk of elective**
Despite being a known clinical fact, the magnitude of this difference in this cohort underscores the importance of rapid triage scoring at emergency admission — not hours later.

---

## Dashboard

The dashboard has three pages:

| Page | Focus |
|---|---|
| Executive overview | KPI cards, mortality by age band and admission type, care unit comparison |
| Risk factor deep dive | Feature importance chart, model metrics, mortality heatmap by age × care unit |
| Temporal patterns | Weekend effect by day of week, shift-level mortality and LOS analysis |

**Note:** The published dashboard displays aggregated statistics only. No patient-level data is included in compliance with the PhysioNet data use agreement (see [Data ethics](#data-ethics--compliance)).

---

## Dataset

**MIMIC-III Clinical Database v1.4**
- Source: [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
- Access: Requires credentialed PhysioNet account (free, requires training completion)
- Coverage: 2001–2012, Beth Israel Deaconess Medical Center, Boston MA
- Size: ~58K ICU stays, 26 linked relational tables
- De-identification: All patient identifiers removed per HIPAA Safe Harbor

**Tables used in this project:**

| Table | Purpose |
|---|---|
| `ADMISSIONS` | Admission metadata, outcome flag, insurance, ethnicity |
| `PATIENTS` | Date of birth, gender, date of death |
| `ICUSTAYS` | ICU-specific stay info, length of stay, care unit |
| `CHARTEVENTS` | Nurse-charted vitals (heart rate, BP, SpO2, temperature, respiratory rate) |
| `LABEVENTS` | Lab results (creatinine, lactate, WBC, haemoglobin, glucose, bicarbonate, bilirubin) |
| `DIAGNOSES_ICD` | ICD-9 diagnosis codes per admission |

---

## Project structure

```
mimic3-icu-analysis/
│
├── README.md
│
├── data/
│   └── .gitkeep               # Raw data not committed (PhysioNet DUA)
│
├── notebooks/
│   ├── 01_cohort_exploration.ipynb    # Initial EDA and cohort validation
│   ├── 02_feature_engineering.ipynb  # Vitals + labs feature extraction
│   └── 03_modelling.ipynb            # Model training, evaluation, SHAP
│
├── scripts/
│   ├── mimic3_icu_cohort.py          # BigQuery cohort extraction
│   ├── mimic3_model.py               # Model training and evaluation
│   └── mimic3_powerbi_export.py      # Aggregated export for Power BI
│
├── outputs/
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── confusion_matrix.png
│   ├── eda_overview.png
│   ├── model_metrics.csv
│   └── powerbi_all_tables.xlsx       # Aggregated — safe to share
│
├── dashboard/
│   └── mimic3_icu_dashboard.pbix     # Power BI Desktop file
│
├── requirements.txt
└── LICENSE
```

> **Important:** The `data/` directory is intentionally empty. Raw MIMIC-III data is never committed to version control — this is required by the PhysioNet data use agreement.

---

## How to reproduce

### Prerequisites

- Python 3.10+
- Google Cloud account with BigQuery access
- PhysioNet credentialed account with MIMIC-III access approved
- Power BI Desktop (free, Windows)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/mimic3-icu-analysis.git
cd mimic3-icu-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

### 4. Run the pipeline

```bash
# Step 1 — Extract cohort from BigQuery (replace project ID in script first)
python scripts/mimic3_icu_cohort.py

# Step 2 — Train and evaluate models
python scripts/mimic3_model.py

# Step 3 — Export aggregated tables for Power BI
python scripts/mimic3_powerbi_export.py
```

### 5. Open the dashboard

Open `dashboard/mimic3_icu_dashboard.pbix` in Power BI Desktop.
Refresh data source to point to your local `outputs/powerbi_all_tables.xlsx`.

---

## Methodology

### Cohort definition
- Adult patients only (age 18–100 at ICU admission)
- First ICU stay per hospitalisation (avoids repeated-measures bias)
- Excluded: neonates (NICU), patients with implausible age values

### Outcome variables
| Variable | Definition |
|---|---|
| `hospital_expire_flag` | In-hospital death (1 = died, 0 = survived) |
| `prolonged_los` | ICU length of stay > 7 days (1 = prolonged, 0 = normal) |

### Feature engineering
- **Vitals:** First-24h min, max, and mean of heart rate, SBP, DBP, SpO2, temperature, respiratory rate (from `CHARTEVENTS`)
- **Labs:** First-24h min, max, and mean of creatinine, lactate, WBC, haemoglobin, glucose, bicarbonate, bilirubin (from `LABEVENTS`)
- **Derived:** Pulse pressure (SBP − DBP), weekend admission flag, night-shift admission flag, emergency admission flag

### Missing data
Clinical data is inherently incomplete. Strategy used:
- **Median imputation** for continuous features (robust to skewed lab distributions)
- **Missingness indicator flags** added as binary features (preserves information about which values were imputed)
- All imputation decisions documented in `02_feature_engineering.ipynb`

### Models

| Model | Rationale |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients map directly to clinical intuition |
| Random Forest | Captures non-linear interactions between clinical features; provides feature importances |

Both models use `class_weight="balanced"` to handle the ~11% positive class rate.
Evaluation uses **5-fold stratified cross-validation** followed by a held-out test set (80/20 split).
Primary metric: **ROC-AUC** (more meaningful than accuracy for imbalanced clinical outcomes).

### Model results

| Model | CV ROC-AUC | Test ROC-AUC | Precision (died) | Recall (died) |
|---|---|---|---|---|
| Logistic Regression | 0.809 ± 0.006 | 0.812 | 0.27 | 0.68 |
| Random Forest | 0.839 ± 0.007 | **0.836** | 0.33 | 0.66 |


---

## Recommendations

Based on the analysis, three evidence-based recommendations for ICU leadership:

**1. Implement a lactate + creatinine early-warning flag**
First-24h lactate and creatinine are the top two modifiable predictors of mortality. A simple threshold-based alert (e.g. lactate > 2.5 mmol/L or creatinine > 2.0 mg/dL within 6h of admission) could trigger automatic specialist consultation for highest-risk patients.

**2. Strengthen weekend and night-shift handoff protocols**
The +1.6 pp mortality gap and 29% longer LOS for night admissions suggest care continuity gaps outside business hours. Structured handoff checklists and on-call specialist response time targets could narrow this gap.

**3. Prioritise elderly MICU patients for early palliative care consultation**
Patients aged 76+ in the MICU face a 26.2% mortality rate. Early, proactive palliative care engagement — rather than reactive — can improve both patient outcomes and ICU resource allocation.

---

## Data ethics & compliance

This project was conducted in full compliance with the [PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciii/view-license/1.4/).

- No raw patient-level data is stored in this repository
- No patient-level data is included in any published dashboard or shared output
- All published outputs use aggregated statistics only
- PhysioNet credentialing and CITI data privacy training completed prior to data access

These practices reflect real-world clinical data governance standards and are noted here intentionally — responsible data handling is as important as analytical skill in healthcare analytics roles.

---

## Requirements

```
google-cloud-bigquery>=3.0
db-dtypes
pyarrow
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
plotly>=5.0
openpyxl>=3.1
shap>=0.43
```

---

## Contact

**Chandraditya Enishetty**
aditya.enishetty@gmail.com

---

*Built as a portfolio project demonstrating end-to-end healthcare data analysis: SQL cohort extraction, Python feature engineering, predictive modelling, and Power BI dashboard delivery.*
