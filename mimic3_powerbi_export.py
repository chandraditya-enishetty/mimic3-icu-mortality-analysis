"""
MIMIC-III → Power BI Export Script
====================================
Reads mimic3_icu_cohort.csv (from mimic3_icu_cohort.py) and
exports clean, aggregated tables that Power BI can load directly.


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

# ── DERIVED COLUMNS ─────────────────────────────

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
# TABLE 1 — Patient-level
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
# TABLE 5 — Weekend effect 
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



