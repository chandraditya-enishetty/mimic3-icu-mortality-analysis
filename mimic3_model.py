"""
MIMIC-III ICU Mortality & LOS — Model Training & Evaluation
============================================================
Reads the cohort CSV produced by mimic3_icu_cohort.py and:
  1. Trains a logistic regression baseline
  2. Trains a random forest classifier
  3. Compares ROC-AUC and precision-recall curves
  4. Plots SHAP feature importances (random forest)
  5. Saves all outputs (plots + metrics CSV) for your portfolio

Run:
    python mimic3_model.py

Outputs:
    roc_curves.png
    feature_importance.png
    shap_summary.png          (if shap installed)
    model_metrics.csv
"""

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration       import calibration_curve

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: install shap for SHAP plots  →  pip install shap")


# ── 2. CONFIG ─────────────────────────────────────────────────────────────────

COHORT_CSV  = "mimic3_icu_cohort.csv"
TARGET      = "hospital_expire_flag"   # swap to "prolonged_los" for LOS analysis
RANDOM_SEED = 42
TEST_SIZE   = 0.2

# Columns to drop before modelling (IDs, raw outcomes, metadata strings)
DROP_COLS = [
    "subject_id", "hadm_id", "icustay_id", "admittime",
    "hospital_expire_flag", "prolonged_los",
    "age_band", "gender", "admission_type",
    "first_careunit", "insurance", "ethnicity",
    "icu_los_days",
]

PALETTE = {
    "lr"  : "#534AB7",   # purple  — logistic regression
    "rf"  : "#1D9E75",   # teal    — random forest
    "ref" : "#B4B2A9",   # gray    — random baseline
}


# ── 3. LOAD & VALIDATE DATA ───────────────────────────────────────────────────

print("=" * 60)
print("MIMIC-III ICU Mortality Model")
print("=" * 60)

df = pd.read_csv(COHORT_CSV)
print(f"\nLoaded              : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Target              : {TARGET}")
print(f"Positive rate       : {df[TARGET].mean():.1%}  (class imbalance note)")

# Drop columns that shouldn't be features
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].copy()
y = df[TARGET].copy()

# Safety check — drop any remaining non-numeric columns
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"Dropping non-numeric cols: {non_numeric}")
    X = X.drop(columns=non_numeric)

print(f"Feature count       : {X.shape[1]}")

# Final missing value check (should be zero after cohort script)
remaining_missing = X.isnull().sum().sum()
if remaining_missing > 0:
    print(f"Warning: {remaining_missing} missing values — filling with median")
    X = X.fillna(X.median())


# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,           # preserve class ratio in both splits
)

print(f"\nTrain set           : {len(X_train):,} patients")
print(f"Test set            : {len(X_test):,} patients")


# ── 5. MODEL DEFINITIONS ─────────────────────────────────────────────────────

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # handles class imbalance
            random_state=RANDOM_SEED,
            C=0.1,                     # L2 regularisation
        )),
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,       # prevents overfitting on clinical data
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ]),
}


# ── 6. CROSS-VALIDATED TRAINING ───────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_results = {}

print("\n── Cross-validation (5-fold, ROC-AUC) ──────────────────────")
for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<25} AUC = {scores.mean():.3f}  ±  {scores.std():.3f}")


# ── 7. FIT ON FULL TRAINING SET ───────────────────────────────────────────────

fitted = {}
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    fitted[name] = pipeline

print("\n── Hold-out test set performance ───────────────────────────")

metrics_rows = []
for name, pipeline in fitted.items():
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)

    print(f"\n  {name}")
    print(f"    ROC-AUC           : {auc:.3f}")
    print(f"    Avg Precision     : {ap:.3f}")
    print(f"    Classification report:")
    report = classification_report(y_test, y_pred, target_names=["Survived", "Died"], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["Survived", "Died"]))

    metrics_rows.append({
        "model"           : name,
        "cv_auc_mean"     : cv_results[name].mean(),
        "cv_auc_std"      : cv_results[name].std(),
        "test_roc_auc"    : auc,
        "test_avg_precision": ap,
        "precision_died"  : report["Died"]["precision"],
        "recall_died"     : report["Died"]["recall"],
        "f1_died"         : report["Died"]["f1-score"],
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv("model_metrics.csv", index=False)
print("\nSaved: model_metrics.csv")


# ── 8. ROC + PRECISION-RECALL CURVES ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"MIMIC-III ICU — Model Comparison (target: {TARGET.replace('_', ' ')})",
    fontsize=13, fontweight="bold",
)

color_map = {
    "Logistic Regression": PALETTE["lr"],
    "Random Forest"      : PALETTE["rf"],
}

# ROC curves
ax = axes[0]
ax.plot([0, 1], [0, 1], color=PALETTE["ref"], linestyle="--", lw=1, label="Random baseline")
for name, pipeline in fitted.items():
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, lw=2, color=color_map[name], label=f"{name}  (AUC = {auc:.3f})")

ax.set_xlabel("False positive rate", fontsize=11)
ax.set_ylabel("True positive rate", fontsize=11)
ax.set_title("ROC curves", fontsize=12)
ax.legend(fontsize=10)

# Precision-recall curves
ax = axes[1]
baseline_pr = y_test.mean()
ax.axhline(baseline_pr, color=PALETTE["ref"], linestyle="--", lw=1, label=f"Random baseline ({baseline_pr:.2f})")
for name, pipeline in fitted.items():
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(rec, prec, lw=2, color=color_map[name], label=f"{name}  (AP = {ap:.3f})")

ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-recall curves", fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: roc_curves.png")


# ── 9. FEATURE IMPORTANCE (RANDOM FOREST) ────────────────────────────────────

rf_pipeline  = fitted["Random Forest"]
rf_clf       = rf_pipeline.named_steps["clf"]
importances  = rf_clf.feature_importances_
feature_names = X.columns.tolist()

imp_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(20)
)

# Clean up feature names for display
imp_df["feature_label"] = (
    imp_df["feature"]
    .str.replace("mean_", "avg ", regex=False)
    .str.replace("min_",  "min ", regex=False)
    .str.replace("max_",  "max ", regex=False)
    .str.replace("_",     " ",    regex=False)
)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(
    imp_df["feature_label"][::-1],
    imp_df["importance"][::-1],
    color=PALETTE["rf"],
    edgecolor="white",
)
ax.set_xlabel("Feature importance (mean decrease in impurity)", fontsize=11)
ax.set_title("Top 20 features — Random Forest\n"
             f"predicting {TARGET.replace('_', ' ')}", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: feature_importance.png")

print("\nTop 10 most important features:")
print(imp_df[["feature_label", "importance"]].head(10).to_string(index=False))


# ── 10. SHAP VALUES (OPTIONAL — requires: pip install shap) ──────────────────

if SHAP_AVAILABLE:
    print("\nComputing SHAP values (random forest — this may take a minute)...")
    explainer  = shap.TreeExplainer(rf_clf)
    # Use a sample of 500 test patients to keep it fast
    X_test_scaled = rf_pipeline.named_steps["scaler"].transform(X_test)
    X_sample = pd.DataFrame(X_test_scaled, columns=feature_names).sample(
        n=min(500, len(X_test)), random_state=RANDOM_SEED
    )
    shap_values = explainer.shap_values(X_sample)

    # shap_values is a list [class0, class1] for classifiers
    shap_vals_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots(figsize=(14, 10))
    shap.summary_plot(
        shap_vals_class1,
        X_sample,
        plot_type="dot",
        max_display=12,
        show=False,
        color=PALETTE["rf"],
    )
    plt.title("SHAP feature importance — impact on mortality prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: shap_summary.png")
else:
    print("\nSkipping SHAP plots — install with: pip install shap")


# ── 11. CONFUSION MATRIX (BEST MODEL) ────────────────────────────────────────

best_name     = metrics_df.sort_values("test_roc_auc", ascending=False).iloc[0]["model"]
best_pipeline = fitted[best_name]
y_pred_best   = best_pipeline.predict(X_test)

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Survived", "Died"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion matrix — {best_name}\n(hold-out test set)", fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: confusion_matrix.png")


# ── 12. PORTFOLIO SUMMARY PRINT ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("PORTFOLIO SUMMARY — copy into your README / report")
print("=" * 60)

best_row = metrics_df.sort_values("test_roc_auc", ascending=False).iloc[0]
print(f"""
Dataset      : MIMIC-III (PhysioNet credentialed)
Cohort       : {len(df):,} adult ICU admissions
Target       : {TARGET.replace('_', ' ')} ({df[TARGET].mean():.1%} positive rate)
Features     : {X.shape[1]} engineered from vitals, labs, and admission metadata
Best model   : {best_row['model']}
  ROC-AUC    : {best_row['test_roc_auc']:.3f}
  Precision  : {best_row['precision_died']:.3f}  (patients predicted to die)
  Recall     : {best_row['recall_died']:.3f}  (actual deaths caught)

Resume bullet (fill in your numbers):
─────────────────────────────────────────────────────────
Analysed {len(df):,} ICU admissions from MIMIC-III (PhysioNet credentialed)
using SQL and Python; engineered {X.shape[1]}+ clinical features from
first-24h vitals and lab events; trained a Random Forest classifier
achieving ROC-AUC {best_row['test_roc_auc']:.3f} in predicting in-hospital mortality,
surfacing top risk factors (age, lactate, creatinine, SpO2) with
SHAP explainability — with recommendations for early-warning triage.
─────────────────────────────────────────────────────────

Output files:
  roc_curves.png         → for your Jupyter notebook & README
  feature_importance.png → for your dashboard / report
  shap_summary.png       → the most impressive visual for interviews
  confusion_matrix.png   → for your report
  model_metrics.csv      → raw numbers table

Next: build mimic3_dashboard.py (Streamlit) or connect to Tableau.
""")
