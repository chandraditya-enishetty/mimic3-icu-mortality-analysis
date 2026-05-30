"""
MIMIC-III ICU Mortality & Length-of-Stay Analysis
==================================================
Portfolio project starter: connects to MIMIC-III on BigQuery,
builds a clean patient cohort, engineers clinical features,
and prepares a model-ready dataframe.

Prerequisites:
    pip install google-cloud-bigquery google-auth pandas numpy
    pip install db-dtypes pyarrow scikit-learn matplotlib seaborn

"""

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.oauth2 import service_account

# ── 2. BIGQUERY CONNECTION ─────────────────────────────────────────────────────

MY_PROJECT = "icu-mortality-los-prediction"   # <-- replace with your GCP project ID
MIMIC_DATASET = "physionet-data.mimiciii_clinical"

client = bigquery.Client(project=MY_PROJECT)

print("BigQuery client initialised.")
print(f"MIMIC-III dataset: {MIMIC_DATASET}\n")


# ── 3. BUILD THE BASE COHORT ───────────────────────────────────────────────────
#
# Rules:
#   - Adult patients only (age 18–100 at admission)
#   - First ICU stay per hospitalisation
#   - Exclude neonates (NICU) and transfers with no ICU time
#
# Outcome variables:
#   - hospital_expire_flag  → in-hospital mortality (0 / 1)
#   - prolonged_los         → ICU LOS > 7 days (0 / 1)

COHORT_QUERY = f"""
WITH first_icu AS (
    -- Keep only the first ICU stay per admission
    SELECT
        hadm_id,
        icustay_id,
        first_careunit,
        los         AS icu_los_days,
        intime      AS icu_intime,
        outtime     AS icu_outtime,
        ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) AS rn
    FROM `{MIMIC_DATASET}.icustays`
),

base AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        fi.icustay_id,
        a.admittime,
        a.dischtime,
        a.admission_type,
        a.insurance,
        a.ethnicity,
        a.hospital_expire_flag,                        -- outcome 1: mortality
        CASE WHEN fi.icu_los_days > 7 THEN 1 ELSE 0
             END                        AS prolonged_los, -- outcome 2: LOS > 7d
        fi.icu_los_days,
        fi.first_careunit,
        fi.icu_intime,
        p.gender,

        -- Age at admission (MIMIC stores DOB shifted; cast carefully)
        DATE_DIFF(
            DATE(a.admittime),
            DATE(p.dob),
            YEAR
        ) AS age

    FROM `{MIMIC_DATASET}.admissions` a
    JOIN `{MIMIC_DATASET}.patients`   p  ON a.subject_id = p.subject_id
    JOIN first_icu                    fi ON a.hadm_id    = fi.hadm_id
    WHERE fi.rn = 1                        -- first ICU stay only
)

SELECT *
FROM base
WHERE age BETWEEN 18 AND 100              -- adults only
  AND first_careunit NOT IN ('NICU')      -- exclude neonates
ORDER BY admittime
"""

print("Fetching base cohort from BigQuery...")
cohort_df = client.query(COHORT_QUERY).to_dataframe()

print(f"Cohort size          : {len(cohort_df):,} ICU admissions")
print(f"Mortality rate       : {cohort_df['hospital_expire_flag'].mean():.1%}")
print(f"Prolonged LOS rate   : {cohort_df['prolonged_los'].mean():.1%}")
print(f"Age range            : {cohort_df['age'].min()}–{cohort_df['age'].max()} years\n")


# ── 4. PULL FIRST-24H VITALS FROM CHARTEVENTS ─────────────────────────────────

VITAL_ITEMIDS = {
    220045: "heart_rate",
    220179: "sbp",
    220180: "dbp",
    220277: "spo2",
    223762: "temperature_c",
    220210: "resp_rate",
}

itemid_list = ", ".join(str(i) for i in VITAL_ITEMIDS)

VITALS_QUERY = f"""
SELECT
    ce.icustay_id,
    ce.itemid,
    ce.valuenum
FROM `{MIMIC_DATASET}.chartevents` ce
INNER JOIN (
    SELECT icustay_id, icu_intime
    FROM UNNEST([
        -- Pass cohort icustay_ids to avoid full table scan
        -- BigQuery will push this predicate down
        STRUCT(0 AS icustay_id, TIMESTAMP '2100-01-01' AS icu_intime)  -- placeholder
    ])
) cohort ON ce.icustay_id = cohort.icustay_id
WHERE ce.itemid IN ({itemid_list})
  AND ce.valuenum IS NOT NULL
  AND ce.valuenum > 0
  AND ce.error IS DISTINCT FROM 1         -- exclude charting errors
  AND ce.charttime BETWEEN cohort.icu_intime
                       AND TIMESTAMP_ADD(cohort.icu_intime, INTERVAL 24 HOUR)
"""

# Efficient approach: pass cohort icustay_ids as a filter
icustay_ids = cohort_df["icustay_id"].dropna().astype(int).tolist()

# BigQuery supports passing arrays — chunk if > 10K stays
CHUNK = 5000
vitals_chunks = []

for i in range(0, len(icustay_ids), CHUNK):
    chunk_ids = icustay_ids[i : i + CHUNK]
    ids_str   = ", ".join(str(x) for x in chunk_ids)

    q = f"""
    SELECT
        ce.icustay_id,
        ce.itemid,
        ce.valuenum
    FROM `{MIMIC_DATASET}.chartevents` ce
    WHERE ce.icustay_id IN ({ids_str})
      AND ce.itemid IN ({itemid_list})
      AND ce.valuenum IS NOT NULL
      AND ce.valuenum > 0
      AND ce.error IS DISTINCT FROM 1
    """
    vitals_chunks.append(client.query(q).to_dataframe())
    print(f"  Fetched vitals chunk {i // CHUNK + 1} / {-(-len(icustay_ids) // CHUNK)}")

vitals_raw = pd.concat(vitals_chunks, ignore_index=True)
print(f"Raw vital sign rows  : {len(vitals_raw):,}\n")


# ── 5. AGGREGATE VITALS TO ONE ROW PER PATIENT ────────────────────────────────

vitals_raw["vital_name"] = vitals_raw["itemid"].map(VITAL_ITEMIDS)

vitals_agg = (
    vitals_raw
    .groupby(["icustay_id", "vital_name"])["valuenum"]
    .agg(["min", "max", "mean"])
    .reset_index()
)

# Pivot to wide format: one column per vital × statistic
vitals_wide = vitals_agg.pivot_table(
    index="icustay_id",
    columns="vital_name",
    values=["min", "max", "mean"],
)
vitals_wide.columns = [f"{stat}_{vital}" for stat, vital in vitals_wide.columns]
vitals_wide = vitals_wide.reset_index()

print(f"Vitals feature columns: {vitals_wide.shape[1] - 1}")


# ── 6. PULL FIRST-24H LAB VALUES ──────────────────────────────────────────────

LAB_ITEMIDS = {
    50912: "creatinine",
    50813: "lactate",
    51301: "wbc",
    51222: "haemoglobin",
    50931: "glucose",
    50882: "bicarbonate",
    50885: "bilirubin",
}
lab_itemid_list = ", ".join(str(i) for i in LAB_ITEMIDS)

lab_chunks = []
for i in range(0, len(icustay_ids), CHUNK):
    chunk_ids = icustay_ids[i : i + CHUNK]
    ids_str   = ", ".join(str(x) for x in chunk_ids)

    q = f"""
    SELECT
        ie.icustay_id,
        le.itemid,
        le.valuenum
    FROM `{MIMIC_DATASET}.labevents` le
    INNER JOIN `{MIMIC_DATASET}.icustays` ie
        ON le.hadm_id = ie.hadm_id
    WHERE ie.icustay_id IN ({ids_str})
      AND le.itemid IN ({lab_itemid_list})
      AND le.valuenum IS NOT NULL
      AND le.valuenum > 0
      AND le.charttime BETWEEN ie.intime
                           AND TIMESTAMP_ADD(ie.intime, INTERVAL 24 HOUR)
    """
    lab_chunks.append(client.query(q).to_dataframe())
    print(f"  Fetched labs chunk {i // CHUNK + 1} / {-(-len(icustay_ids) // CHUNK)}")

labs_raw = pd.concat(lab_chunks, ignore_index=True)
labs_raw["lab_name"] = labs_raw["itemid"].map(LAB_ITEMIDS)

labs_agg = (
    labs_raw
    .groupby(["icustay_id", "lab_name"])["valuenum"]
    .agg(["min", "max", "mean"])
    .reset_index()
)
labs_wide = labs_agg.pivot_table(
    index="icustay_id",
    columns="lab_name",
    values=["min", "max", "mean"],
)
labs_wide.columns = [f"{stat}_{lab}" for stat, lab in labs_wide.columns]
labs_wide = labs_wide.reset_index()

print(f"\nLabs feature columns : {labs_wide.shape[1] - 1}")


# ── 7. MERGE INTO MASTER DATAFRAME ────────────────────────────────────────────

df = (
    cohort_df
    .merge(vitals_wide, on="icustay_id", how="left")
    .merge(labs_wide,   on="icustay_id", how="left")
)

print(f"\nMaster dataframe     : {df.shape[0]:,} rows × {df.shape[1]} columns")


# ── 8. FEATURE ENGINEERING ────────────────────────────────────────────────────

# Encode categoricals
df["gender_enc"]          = (df["gender"] == "M").astype(int)
df["emergency_admission"] = (df["admission_type"] == "EMERGENCY").astype(int)
df["weekend_admission"]   = pd.to_datetime(df["admittime"]).dt.dayofweek.isin([5, 6]).astype(int)
df["night_admission"]     = pd.to_datetime(df["admittime"]).dt.hour.between(0, 7).astype(int)

# Age bands (useful for EDA)
df["age_band"] = pd.cut(
    df["age"],
    bins=[17, 40, 60, 75, 100],
    labels=["18–40", "41–60", "61–75", "76+"],
)

# Pulse pressure (SBP - DBP) — clinical feature
if "mean_sbp" in df.columns and "mean_dbp" in df.columns:
    df["pulse_pressure"] = df["mean_sbp"] - df["mean_dbp"]

print("Feature engineering  : done")


# ── 9. HANDLE MISSING VALUES ──────────────────────────────────────────────────

# median imputation

feature_cols = (
    [c for c in df.columns if c.startswith(("min_", "max_", "mean_"))]
    + ["pulse_pressure"]
)
feature_cols = [c for c in feature_cols if c in df.columns]

missing_summary = df[feature_cols].isnull().mean().sort_values(ascending=False)
print("\nMissing data summary (top 10):")
print(missing_summary.head(10).apply(lambda x: f"{x:.1%}").to_string())

# Median imputation
for col in feature_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Add missingness indicator flags 
for col in feature_cols:
    df[f"{col}_missing"] = df[col].isnull().astype(int)

print("\nMissing value imputation: complete (median strategy)")


# ── 10. QUICK EDA PLOTS ───────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("MIMIC-III ICU Cohort — Exploratory Analysis", fontsize=14, fontweight="bold")

# Plot 1: Mortality rate by age band
mort_by_age = df.groupby("age_band", observed=True)["hospital_expire_flag"].mean() * 100
axes[0, 0].bar(mort_by_age.index.astype(str), mort_by_age.values, color="#1D9E75", edgecolor="white")
axes[0, 0].set_title("In-hospital mortality rate by age band", fontsize=11)
axes[0, 0].set_ylabel("Mortality rate (%)")
axes[0, 0].set_xlabel("Age band")

# Plot 2: ICU LOS distribution
axes[0, 1].hist(df["icu_los_days"].clip(upper=30), bins=40, color="#534AB7", edgecolor="white")
axes[0, 1].axvline(7, color="#E24B4A", linestyle="--", label="7-day threshold")
axes[0, 1].set_title("ICU length of stay distribution", fontsize=11)
axes[0, 1].set_xlabel("LOS (days, capped at 30)")
axes[0, 1].set_ylabel("Count")
axes[0, 1].legend()

# Plot 3: Mortality by admission type
mort_by_type = df.groupby("admission_type")["hospital_expire_flag"].mean() * 100
axes[1, 0].barh(mort_by_type.index, mort_by_type.values, color="#D85A30", edgecolor="white")
axes[1, 0].set_title("Mortality rate by admission type", fontsize=11)
axes[1, 0].set_xlabel("Mortality rate (%)")

# Plot 4: Weekend effect on LOS
weekend_los = df.groupby("weekend_admission")["icu_los_days"].mean()
axes[1, 1].bar(
    ["Weekday", "Weekend"],
    weekend_los.values,
    color=["#1D9E75", "#BA7517"],
    edgecolor="white",
)
axes[1, 1].set_title("Mean ICU LOS: weekday vs weekend admission", fontsize=11)
axes[1, 1].set_ylabel("Mean LOS (days)")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nEDA plot saved: eda_overview.png")


# ── 11. EXPORT MODEL-READY DATASET ────────────────────────────────────────────

OUTCOME_COLS   = ["hospital_expire_flag", "prolonged_los"]
ID_COLS        = ["subject_id", "hadm_id", "icustay_id", "admittime"]
METADATA_COLS  = ["age", "age_band", "gender", "admission_type",
                  "first_careunit", "icu_los_days", "insurance", "ethnicity"]
ENGINEERED     = ["gender_enc", "emergency_admission", "weekend_admission",
                  "night_admission", "pulse_pressure"]

model_cols = ID_COLS + OUTCOME_COLS + METADATA_COLS + ENGINEERED + feature_cols
model_df   = df[[c for c in model_cols if c in df.columns]].copy()

model_df.to_csv("mimic3_icu_cohort.csv", index=False)

print(f"\nExported: mimic3_icu_cohort.csv")
print(f"  Shape  : {model_df.shape[0]:,} rows × {model_df.shape[1]} columns")
print(f"  Outcomes: mortality={model_df['hospital_expire_flag'].mean():.1%}, "
      f"prolonged_los={model_df['prolonged_los'].mean():.1%}")
print("\nNext step: run mimic3_model.py to train and evaluate classifiers.")
