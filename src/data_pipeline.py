import os
import pandas as pd


TARGET_COLS = [
    "Hypertension",
    "Hyperlipidemia",
    "Diabetes",
    "Atrial_Fibrillation",
]

NUMERIC_FEATURES = [
    "age",
    "Potassium",
    "Sodium",
    "Heart_Rate",
    "Respiratory_Rate",
]


def map_disease(icd_code, icd_version):
    """
    Convert detailed ICD-9 and ICD-10 diagnosis codes into
    the four disease groups used by this research project.
    """
    if pd.isna(icd_code):
        return None

    code = str(icd_code).replace(".", "").strip().upper()

    try:
        version = int(icd_version)
    except (TypeError, ValueError):
        version = None

    if version == 9:
        if code.startswith("401"):
            return "Hypertension"

        if code.startswith("272"):
            return "Hyperlipidemia"

        if code.startswith("250"):
            return "Diabetes"

        if code.startswith("42731"):
            return "Atrial_Fibrillation"

    elif version == 10:
        if code.startswith(("I10", "I11", "I12", "I13", "I15")):
            return "Hypertension"

        if code.startswith("E78"):
            return "Hyperlipidemia"

        if code.startswith(("E08", "E09", "E10", "E11", "E13")):
            return "Diabetes"

        if code.startswith("I48"):
            return "Atrial_Fibrillation"

    return None


def preprocess_ehr_pipeline(
    raw_dir="data/raw",
    output_path="data/processed/ehr_unscaled_dataset.csv",
):
    print("--- Step 1: Cohort Selection & Filtering ---")

    patients = pd.read_csv(
        os.path.join(raw_dir, "patients.csv")
    )

    admissions = pd.read_csv(
        os.path.join(raw_dir, "admissions.csv")
    )

    cohort = pd.merge(
        admissions,
        patients,
        on="subject_id",
        how="inner",
    )

    cohort["admittime"] = pd.to_datetime(
        cohort["admittime"],
        errors="coerce",
    )

    cohort["dischtime"] = pd.to_datetime(
        cohort["dischtime"],
        errors="coerce",
    )

    cohort = cohort.dropna(
        subset=["admittime", "dischtime"]
    ).copy()

    cohort["admission_year"] = cohort["admittime"].dt.year

    cohort["age"] = (
        cohort["anchor_age"]
        + cohort["admission_year"]
        - cohort["anchor_year"]
    )

    cohort = cohort[cohort["age"] >= 18].copy()

    print(f"Total valid adult admissions: {len(cohort)}")
    print(
        f"Total unique adult patients: "
        f"{cohort['subject_id'].nunique()}"
    )

    print("\n--- Step 2: Clinical Grouping & Target Encoding ---")

    diagnoses = pd.read_csv(
        os.path.join(raw_dir, "diagnoses_icd.csv"),
        dtype={"icd_code": str},
    )

    diagnoses["disease"] = diagnoses.apply(
        lambda row: map_disease(
            row["icd_code"],
            row.get("icd_version"),
        ),
        axis=1,
    )

    mapped_dx = diagnoses.dropna(
        subset=["disease"]
    ).copy()

    disease_matrix = pd.crosstab(
        mapped_dx["hadm_id"],
        mapped_dx["disease"],
    )

    disease_matrix = (
        disease_matrix
        .reindex(columns=TARGET_COLS, fill_value=0)
        .gt(0)
        .astype(int)
        .reset_index()
    )

    cohort = pd.merge(
        cohort[
            [
                "subject_id",
                "hadm_id",
                "gender",
                "age",
            ]
        ],
        disease_matrix,
        on="hadm_id",
        how="left",
    )

    cohort[TARGET_COLS] = (
        cohort[TARGET_COLS]
        .fillna(0)
        .astype(int)
    )

    cohort["gender_encoded"] = (
        cohort["gender"]
        .astype(str)
        .str.upper()
        .eq("M")
        .astype(int)
    )

    print("Target distribution:")
    print(cohort[TARGET_COLS].sum())

    print("\n--- Step 3: Feature Extraction ---")

    labs = pd.read_csv(
        os.path.join(raw_dir, "labevents.csv"),
        usecols=["hadm_id", "itemid", "valuenum"],
    )

    charts = pd.read_csv(
        os.path.join(raw_dir, "chartevents.csv"),
        usecols=["hadm_id", "itemid", "valuenum"],
    )

    lab_map = {
        50971: "Potassium",
        50983: "Sodium",
    }

    labs_filtered = labs[
        labs["itemid"].isin(lab_map)
    ].copy()

    labs_filtered["feature"] = (
        labs_filtered["itemid"].map(lab_map)
    )

    labs_filtered["valuenum"] = pd.to_numeric(
        labs_filtered["valuenum"],
        errors="coerce",
    )

    lab_summary = (
        labs_filtered
        .groupby(["hadm_id", "feature"])["valuenum"]
        .mean()
        .unstack()
        .reset_index()
    )

    chart_map = {
        220045: "Heart_Rate",
        220210: "Respiratory_Rate",
    }

    charts_filtered = charts[
        charts["itemid"].isin(chart_map)
    ].copy()

    charts_filtered["feature"] = (
        charts_filtered["itemid"].map(chart_map)
    )

    charts_filtered["valuenum"] = pd.to_numeric(
        charts_filtered["valuenum"],
        errors="coerce",
    )

    chart_summary = (
        charts_filtered
        .groupby(["hadm_id", "feature"])["valuenum"]
        .mean()
        .unstack()
        .reset_index()
    )

    cohort = pd.merge(
        cohort,
        lab_summary,
        on="hadm_id",
        how="left",
    )

    cohort = pd.merge(
        cohort,
        chart_summary,
        on="hadm_id",
        how="left",
    )

    print("\n--- Step 4: Preserve Raw Features ---")

    for column in NUMERIC_FEATURES:
        cohort[column] = pd.to_numeric(
            cohort[column],
            errors="coerce",
        )

    print("Missing values before train-only imputation:")
    print(cohort[NUMERIC_FEATURES].isna().sum())

    feature_cols = [
        "gender_encoded",
        "age",
        "Potassium",
        "Sodium",
        "Heart_Rate",
        "Respiratory_Rate",
    ]

    final_output = cohort[
        ["subject_id", "hadm_id"]
        + feature_cols
        + TARGET_COLS
    ].copy()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    final_output.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved unscaled dataset to: {output_path}")
    print(f"Shape: {final_output.shape}")
    print(f"Unique patients: {final_output['subject_id'].nunique()}")

    return final_output


if __name__ == "__main__":
    preprocess_ehr_pipeline()