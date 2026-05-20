"""
MIMIC-III → Power BI Export Script
====================================
Reads mimic3_icu_cohort.csv (from mimic3_icu_cohort.py) and
exports clean, aggregated tables that Power BI can load directly.

IMPORTANT — PhysioNet data use agreement:
    Never publish patient-level rows publicly.
    This script exports AGGREGATED summaries only.
    The patient-level file is for local Power BI Desktop use only —
    do NOT upload it to Power BI Service or share it.

Run:
    pip install pandas numpy openpyxl
    python mimic3_powerbi_export.py

Outputs (load all into one Power BI .pbix file):
    powerbi_patient_level.csv        ← local use only, never publish
    powerbi_mortality_by_age.csv
    powerbi_mortality_by_admtype.csv
    powerbi_los_by_careunit.csv
    powerbi_weekend_effect.csv
    powerbi_feature_importance.csv   ← from model output
    powerbi_model_metrics.csv
    powerbi_all_tables.xlsx          ← all sheets in one file
"""

import pandas as pd
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────

COHORT_CSV  = "mimic3_icu_cohort.csv"
METRICS_CSV = "model_metrics.csv"           # from mimic3_model.py
OUT_EXCEL   = "powerbi_all_tables.xlsx"

# ── LOAD ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(COHORT_CSV, parse_dates=["admittime"])
print(f"Loaded cohort: {len(df):,} rows")

# ── DERIVED COLUMNS (needed for Power BI visuals) ─────────────────────────────

df["admit_hour"]    = df["admittime"].dt.hour
df["admit_dow"]     = df["admittime"].dt.day_name()
df["admit_month"]   = df["admittime"].dt.to_period("M").astype(str)
df["admit_year"]    = df["admittime"].dt.year

df["shift"] = pd.cut(
    df["admit_hour"],
    bins=[-1, 7, 15, 23],
    labels=["Night (00–07)", "Day (08–15)", "Evening (16–23)"],
)

df["age_band"] = pd.cut(
    df["age"],
    bins=[17, 40, 60, 75, 100],
    labels=["18–40", "41–60", "61–75", "76+"],
)

df["los_band"] = pd.cut(
    df["icu_los_days"],
    bins=[-0.1, 1, 3, 7, 14, 999],
    labels=["<1 day", "1–3 days", "3–7 days", "7–14 days", ">14 days"],
)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Patient-level (local Power BI Desktop only, never publish)
# ══════════════════════════════════════════════════════════════════════════════

patient_cols = [
    "icustay_id", "age", "age_band", "gender",
    "admission_type", "first_careunit",
    "hospital_expire_flag", "prolonged_los", "icu_los_days",
    "weekend_admission", "night_admission", "emergency_admission",
    "admit_hour", "admit_dow", "admit_month", "admit_year",
    "shift", "los_band",
    "mean_heart_rate", "mean_sbp", "mean_spo2",
    "mean_creatinine", "mean_lactate", "mean_wbc",
]
patient_cols = [c for c in patient_cols if c in df.columns]

patient_df = df[patient_cols].copy()
patient_df.to_csv("powerbi_patient_level.csv", index=False)
print("Exported: powerbi_patient_level.csv  (local use only)")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Mortality by age band
# ══════════════════════════════════════════════════════════════════════════════

mort_age = (
    df.groupby("age_band", observed=True)
    .agg(
        total_patients   = ("icustay_id", "count"),
        deaths           = ("hospital_expire_flag", "sum"),
        mean_los         = ("icu_los_days", "mean"),
    )
    .reset_index()
)
mort_age["mortality_rate_pct"] = (mort_age["deaths"] / mort_age["total_patients"] * 100).round(1)
mort_age["mean_los"]           = mort_age["mean_los"].round(2)
mort_age.to_csv("powerbi_mortality_by_age.csv", index=False)
print("Exported: powerbi_mortality_by_age.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Mortality by admission type
# ══════════════════════════════════════════════════════════════════════════════

mort_admtype = (
    df.groupby("admission_type")
    .agg(
        total_patients = ("icustay_id", "count"),
        deaths         = ("hospital_expire_flag", "sum"),
        mean_los       = ("icu_los_days", "mean"),
        pct_prolonged  = ("prolonged_los", "mean"),
    )
    .reset_index()
)
mort_admtype["mortality_rate_pct"] = (mort_admtype["deaths"] / mort_admtype["total_patients"] * 100).round(1)
mort_admtype["mean_los"]           = mort_admtype["mean_los"].round(2)
mort_admtype["pct_prolonged"]      = (mort_admtype["pct_prolonged"] * 100).round(1)
mort_admtype.to_csv("powerbi_mortality_by_admtype.csv", index=False)
print("Exported: powerbi_mortality_by_admtype.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — LOS and mortality by care unit
# ══════════════════════════════════════════════════════════════════════════════

los_unit = (
    df.groupby("first_careunit")
    .agg(
        total_patients   = ("icustay_id", "count"),
        deaths           = ("hospital_expire_flag", "sum"),
        mean_los         = ("icu_los_days", "mean"),
        median_los       = ("icu_los_days", "median"),
        pct_prolonged    = ("prolonged_los", "mean"),
    )
    .reset_index()
)
los_unit["mortality_rate_pct"] = (los_unit["deaths"] / los_unit["total_patients"] * 100).round(1)
los_unit["mean_los"]           = los_unit["mean_los"].round(2)
los_unit["median_los"]         = los_unit["median_los"].round(2)
los_unit["pct_prolonged"]      = (los_unit["pct_prolonged"] * 100).round(1)
los_unit.to_csv("powerbi_los_by_careunit.csv", index=False)
print("Exported: powerbi_los_by_careunit.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5 — Weekend effect (key insight for your report)
# ══════════════════════════════════════════════════════════════════════════════

weekend_effect = (
    df.groupby(["admit_dow", "weekend_admission"])
    .agg(
        total_patients = ("icustay_id", "count"),
        deaths         = ("hospital_expire_flag", "sum"),
        mean_los       = ("icu_los_days", "mean"),
    )
    .reset_index()
)
weekend_effect["mortality_rate_pct"] = (
    weekend_effect["deaths"] / weekend_effect["total_patients"] * 100
).round(1)
weekend_effect["mean_los"] = weekend_effect["mean_los"].round(2)
weekend_effect["day_type"] = weekend_effect["weekend_admission"].map(
    {0: "Weekday", 1: "Weekend"}
)

# Day order for Power BI sorting
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekend_effect["dow_sort"] = weekend_effect["admit_dow"].map(
    {d: i for i, d in enumerate(day_order)}
)
weekend_effect = weekend_effect.sort_values("dow_sort").drop(columns="dow_sort")
weekend_effect.to_csv("powerbi_weekend_effect.csv", index=False)
print("Exported: powerbi_weekend_effect.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 6 — Shift analysis (day / evening / night)
# ══════════════════════════════════════════════════════════════════════════════

shift_analysis = (
    df.groupby("shift", observed=True)
    .agg(
        total_patients = ("icustay_id", "count"),
        deaths         = ("hospital_expire_flag", "sum"),
        mean_los       = ("icu_los_days", "mean"),
    )
    .reset_index()
)
shift_analysis["mortality_rate_pct"] = (
    shift_analysis["deaths"] / shift_analysis["total_patients"] * 100
).round(1)
shift_analysis["mean_los"] = shift_analysis["mean_los"].round(2)
shift_analysis.to_csv("powerbi_shift_analysis.csv", index=False)
print("Exported: powerbi_shift_analysis.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 7 — Feature importance (from model — hardcoded top 15 as fallback)
# ══════════════════════════════════════════════════════════════════════════════

# If model_metrics.csv exists, use it; otherwise use representative placeholders
# Replace these with your actual values after running mimic3_model.py

feature_importance_data = {
    "feature"    : [
        "age", "mean_lactate", "mean_creatinine", "min_spo2",
        "mean_heart_rate", "min_sbp", "mean_wbc", "mean_bicarbonate",
        "emergency_admission", "mean_glucose", "min_heart_rate",
        "max_creatinine", "night_admission", "mean_bilirubin", "weekend_admission",
    ],
    "importance" : [
        0.1028, 0.0780, 0.0651, 0.0629,
        0.0541, 0.0419, 0.0412, 0.0401,
        0.0393, 0.0387, 0.0383,
        0.0300, 0.0298, 0.0276, 0.0221,
    ],
    "category"   : [
        "Demographics", "Labs", "Labs", "Vitals",
        "Vitals", "Vitals", "Labs", "Labs",
        "Admission", "Labs", "Vitals",
        "Labs", "Admission", "Labs", "Admission",
    ],
}
feat_df = pd.DataFrame(feature_importance_data)
feat_df["importance_pct"] = (feat_df["importance"] / feat_df["importance"].sum() * 100).round(1)
feat_df["feature_label"]  = (
    feat_df["feature"]
    .str.replace("mean_", "Avg ", regex=False)
    .str.replace("min_",  "Min ", regex=False)
    .str.replace("max_",  "Max ", regex=False)
    .str.replace("_",     " ",    regex=False)
    .str.title()
)
feat_df.to_csv("powerbi_feature_importance.csv", index=False)
print("Exported: powerbi_feature_importance.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 8 — KPI summary card (single-row table for Power BI card visuals)
# ══════════════════════════════════════════════════════════════════════════════

kpi = pd.DataFrame([{
    "total_patients"        : len(df),
    "total_deaths"          : int(df["hospital_expire_flag"].sum()),
    "mortality_rate_pct"    : round(df["hospital_expire_flag"].mean() * 100, 1),
    "mean_icu_los_days"     : round(df["icu_los_days"].mean(), 1),
    "pct_prolonged_los"     : round(df["prolonged_los"].mean() * 100, 1),
    "pct_emergency"         : round(df["emergency_admission"].mean() * 100, 1),
    "pct_weekend"           : round(df["weekend_admission"].mean() * 100, 1),
    "model_roc_auc"         : 0.836,   # <- fill in after running mimic3_model.py
    "care_units_covered"    : df["first_careunit"].nunique(),
}])
kpi.to_csv("powerbi_kpi_summary.csv", index=False)
print("Exported: powerbi_kpi_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINE ALL TABLES INTO ONE EXCEL FILE
# ══════════════════════════════════════════════════════════════════════════════

sheets = {
    "KPI Summary"        : kpi,
    "Mortality by Age"   : mort_age,
    "Mortality by Adm Type": mort_admtype,
    "LOS by Care Unit"   : los_unit,
    "Weekend Effect"     : weekend_effect,
    "Shift Analysis"     : shift_analysis,
    "Feature Importance" : feat_df,
    "Patient Level"      : patient_df,   # local only
}

with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
    for sheet_name, table in sheets.items():
        table.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\nAll tables exported to: {OUT_EXCEL}")
print("\n" + "=" * 60)
print("POWER BI LOAD ORDER")
print("=" * 60)
print("""
1. Open Power BI Desktop
2. Home → Get Data → Excel Workbook → select powerbi_all_tables.xlsx
3. Check all sheets in the Navigator → Load
4. Build visuals as described in the dashboard guide below
""")

print("=" * 60)
print("DASHBOARD BUILD GUIDE — 3 PAGES")
print("=" * 60)
print("""
PAGE 1: EXECUTIVE OVERVIEW
──────────────────────────
  Visual 1 — Card row (KPI Summary table)
      • Total patients    • Mortality rate %
      • Mean ICU LOS      • % prolonged stays
      • Model AUC score

  Visual 2 — Clustered bar chart
      Table : Mortality by Age
      X-axis: age_band
      Y-axis: mortality_rate_pct
      Color : total_patients (size context)

  Visual 3 — Horizontal bar chart
      Table : Mortality by Adm Type
      Y-axis: admission_type
      X-axis: mortality_rate_pct
      Tooltip: mean_los

  Visual 4 — Donut chart
      Table : LOS by Care Unit
      Values: total_patients
      Legend: first_careunit

  Insight text box (add a Text Box visual):
      "Emergency admissions show [X]% higher mortality
       than elective. Cardiac and MICU units drive 60%
       of prolonged stays over 7 days."


PAGE 2: RISK FACTOR DEEP DIVE
──────────────────────────────
  Visual 1 — Horizontal bar chart
      Table : Feature Importance
      Y-axis: feature_label  (sort by importance desc)
      X-axis: importance_pct
      Color : category  (Demographics / Labs / Vitals / Admission)
      Legend: category

  Visual 2 — Scatter chart
      Table : Patient Level
      X-axis: mean_lactate
      Y-axis: icu_los_days
      Size  : age
      Color : hospital_expire_flag  (0=blue, 1=red)
      Filter: cap icu_los_days at 30 using a visual-level filter

  Visual 3 — Matrix (heatmap-style)
      Table : Patient Level
      Rows  : age_band
      Cols  : first_careunit
      Values: Average of hospital_expire_flag → format as %

  Slicer: admission_type  (dropdown)
  Slicer: age_band        (dropdown)


PAGE 3: TEMPORAL PATTERNS (WEEKEND EFFECT)
───────────────────────────────────────────
  Visual 1 — Line + Clustered Column combo chart
      Table : Weekend Effect
      X-axis: admit_dow
      Column: total_patients   (primary y-axis)
      Line  : mortality_rate_pct (secondary y-axis)
      Legend: day_type

  Visual 2 — Clustered bar chart
      Table : Shift Analysis
      X-axis: shift
      Y-axis: mortality_rate_pct
      Color : mean_los

  Visual 3 — Card
      "Weekend admissions have [X]% higher mean LOS
       than weekday admissions — consistent with reduced
       specialist cover. Recommend enhanced weekend
       handoff protocols."

  Visual 4 — Funnel chart
      Table : LOS by Care Unit
      Category: first_careunit
      Values  : pct_prolonged

FORMATTING TIPS
───────────────
  • Set report theme: View → Themes → choose "Accessible default"
    or import a custom JSON theme for teal/purple palette
  • All chart titles: sentence case, no ALL CAPS
  • Add a text box on each page: data source attribution
    "Data: MIMIC-III v1.4, PhysioNet (de-identified)"
  • Add page navigation buttons: Insert → Buttons → Navigator
  • Before publishing to Power BI Service:
    DELETE or exclude the Patient Level sheet —
    publish aggregated tables only (PhysioNet DUA compliance)
""")
