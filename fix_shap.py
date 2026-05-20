import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Load cohort ───────────────────────────────────────────
df = pd.read_csv("mimic3_icu_cohort.csv")

DROP_COLS = [
    "subject_id", "hadm_id", "icustay_id", "admittime",
    "hospital_expire_flag", "prolonged_los",
    "age_band", "gender", "admission_type",
    "first_careunit", "insurance", "ethnicity", "icu_los_days",
]

TARGET = "hospital_expire_flag"
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
X = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Fit model ─────────────────────────────────────────────
rf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_leaf=20, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )),
])
print("Fitting model...")
rf.fit(X_train, y_train)

# ── SHAP ──────────────────────────────────────────────────
print("Computing SHAP values...")
clf    = rf.named_steps["clf"]
scaler = rf.named_steps["scaler"]

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns,
).sample(n=500, random_state=42)

explainer  = shap.TreeExplainer(clf)
shap_values = explainer(X_test_scaled)

# Binary classifier — take class 1 (mortality)
vals = shap_values[..., 1]

plt.figure(figsize=(14, 10))
shap.plots.beeswarm(vals, max_display=12, show=False)
plt.title(
    "SHAP feature importance — impact on ICU mortality prediction",
    fontsize=13, pad=20,
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Done — shap_summary.png saved.")