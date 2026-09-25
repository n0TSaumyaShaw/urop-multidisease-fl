from pathlib import Path
import sys

import numpy as np
import pandas as pd


DATA_PATH = Path("data/processed/ehr_unscaled_dataset.csv")
SPLITS_PATH = Path("data/processed/patient_splits.csv")

ID_COLUMNS = [
    "subject_id",
    "hadm_id",
]

FEATURE_COLUMNS = [
    "gender_encoded",
    "age",
    "Potassium",
    "Sodium",
    "Heart_Rate",
    "Respiratory_Rate",
]

TARGET_COLUMNS = [
    "Hypertension",
    "Hyperlipidemia",
    "Diabetes",
    "Atrial_Fibrillation",
]

REQUIRED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


def print_heading(title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def validate_dataset():
    errors = []
    warnings = []

    print_heading("EHR DATASET VALIDATION")

    # ---------------------------------------------------------
    # Check 1: Dataset file
    # ---------------------------------------------------------
    print("\n[1] Checking dataset file")

    if not DATA_PATH.exists():
        print(f"FAIL: Dataset not found: {DATA_PATH}")
        return False

    print(f"PASS: Dataset found: {DATA_PATH}")

    try:
        data = pd.read_csv(DATA_PATH)
    except Exception as error:
        print(f"FAIL: Dataset could not be loaded: {error}")
        return False

    print(f"Admissions: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    # ---------------------------------------------------------
    # Check 2: Required columns
    # ---------------------------------------------------------
    print("\n[2] Checking required columns")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing required columns: {missing_columns}"
        )
        print(f"FAIL: Missing columns: {missing_columns}")
        return False

    print("PASS: All required columns are available")

    # ---------------------------------------------------------
    # Check 3: Patient and admission identifiers
    # ---------------------------------------------------------
    print("\n[3] Checking patient and admission identifiers")

    unique_patients = data["subject_id"].nunique()
    unique_admissions = data["hadm_id"].nunique()

    print(f"Unique patients: {unique_patients}")
    print(f"Unique admissions: {unique_admissions}")

    missing_subject_ids = data["subject_id"].isna().sum()
    missing_admission_ids = data["hadm_id"].isna().sum()

    if missing_subject_ids > 0:
        errors.append(
            f"{missing_subject_ids} rows have missing subject_id"
        )

    if missing_admission_ids > 0:
        errors.append(
            f"{missing_admission_ids} rows have missing hadm_id"
        )

    duplicate_admissions = data["hadm_id"].duplicated().sum()

    if duplicate_admissions > 0:
        errors.append(
            f"{duplicate_admissions} duplicate admission IDs found"
        )
        print(
            f"FAIL: Duplicate admission IDs: "
            f"{duplicate_admissions}"
        )
    else:
        print("PASS: Every admission ID is unique")

    # ---------------------------------------------------------
    # Check 4: Missing values
    # ---------------------------------------------------------
    print("\n[4] Checking missing values")

    missing_values = data[REQUIRED_COLUMNS].isna().sum()
    missing_percentages = (
        data[REQUIRED_COLUMNS].isna().mean() * 100
    )

    missing_report = pd.DataFrame({
        "missing_count": missing_values,
        "missing_percentage": missing_percentages.round(2),
    })

    print(missing_report.to_string())

    columns_with_missing_values = missing_values[
        missing_values > 0
    ]

    if not columns_with_missing_values.empty:
        warnings.append(
            "Missing clinical values were found. These must be "
            "imputed using training data only."
        )

    # Targets and identifiers must never be missing
    protected_columns = ID_COLUMNS + TARGET_COLUMNS
    protected_missing = data[protected_columns].isna().sum()

    if protected_missing.sum() > 0:
        errors.append(
            "Missing values were found in identifier or target columns"
        )

    # ---------------------------------------------------------
    # Check 5: Infinite feature values
    # ---------------------------------------------------------
    print("\n[5] Checking infinite values")

    numeric_features = data[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    infinite_count = np.isinf(
        numeric_features.to_numpy(dtype=float)
    ).sum()

    if infinite_count > 0:
        errors.append(
            f"{infinite_count} infinite feature values found"
        )
        print(f"FAIL: Infinite values found: {infinite_count}")
    else:
        print("PASS: No infinite feature values found")

    # ---------------------------------------------------------
    # Check 6: Demographic values
    # ---------------------------------------------------------
    print("\n[6] Checking demographic values")

    invalid_gender = ~data["gender_encoded"].isin([0, 1])

    if invalid_gender.any():
        errors.append(
            f"{invalid_gender.sum()} invalid gender values found"
        )
        print(
            f"FAIL: Invalid gender values: "
            f"{invalid_gender.sum()}"
        )
    else:
        print("PASS: gender_encoded contains only 0 and 1")

    invalid_age = (
        data["age"].notna()
        & ~data["age"].between(18, 120)
    )

    if invalid_age.any():
        errors.append(
            f"{invalid_age.sum()} ages are outside 18–120"
        )
        print(f"FAIL: Invalid ages: {invalid_age.sum()}")
    else:
        print("PASS: All available ages are between 18 and 120")

    # ---------------------------------------------------------
    # Check 7: Disease targets
    # ---------------------------------------------------------
    print("\n[7] Checking disease targets")

    for disease in TARGET_COLUMNS:
        invalid_target = ~data[disease].isin([0, 1])

        if invalid_target.any():
            errors.append(
                f"{disease} contains non-binary values"
            )
            print(f"FAIL: {disease} contains invalid values")
            continue

        positive_cases = int(data[disease].sum())
        negative_cases = int(len(data) - positive_cases)

        print(
            f"{disease}: "
            f"{positive_cases} positive, "
            f"{negative_cases} negative"
        )

        if positive_cases == 0:
            errors.append(
                f"{disease} has no positive cases"
            )

        if negative_cases == 0:
            errors.append(
                f"{disease} has no negative cases"
            )

    # ---------------------------------------------------------
    # Check 8: Approximate clinical ranges
    # ---------------------------------------------------------
    print("\n[8] Checking approximate clinical ranges")

    clinical_ranges = {
        "Potassium": (1.5, 8.0),
        "Sodium": (100, 180),
        "Heart_Rate": (20, 250),
        "Respiratory_Rate": (5, 80),
    }

    for feature, (minimum, maximum) in clinical_ranges.items():
        outside_range = (
            data[feature].notna()
            & ~data[feature].between(minimum, maximum)
        )

        count = int(outside_range.sum())

        if count > 0:
            warnings.append(
                f"{feature}: {count} values outside the "
                f"approximate range {minimum}–{maximum}"
            )
            print(
                f"WARNING: {feature} has {count} values outside "
                f"{minimum}–{maximum}"
            )
        else:
            print(f"PASS: {feature} values are within range")

    # ---------------------------------------------------------
    # Check 9: Patient split leakage
    # ---------------------------------------------------------
    print("\n[9] Checking patient split leakage")

    if not SPLITS_PATH.exists():
        warnings.append(
            f"Patient split file not found: {SPLITS_PATH}"
        )
        print("WARNING: Patient split file was not found")
    else:
        splits = pd.read_csv(SPLITS_PATH)

        split_column = None

        for candidate in ["split", "dataset_split", "set"]:
            if candidate in splits.columns:
                split_column = candidate
                break

        if "subject_id" not in splits.columns:
            errors.append(
                "patient_splits.csv does not contain subject_id"
            )
            print("FAIL: subject_id is missing from split file")

        elif split_column is None:
            errors.append(
                "No split column found in patient_splits.csv"
            )
            print("FAIL: Split-name column could not be identified")

        else:
            patient_split_counts = (
                splits.groupby("subject_id")[split_column]
                .nunique()
            )

            leaked_patients = patient_split_counts[
                patient_split_counts > 1
            ]

            if len(leaked_patients) > 0:
                errors.append(
                    f"{len(leaked_patients)} patients appear in "
                    "multiple splits"
                )
                print(
                    f"FAIL: {len(leaked_patients)} patients appear "
                    "in multiple splits"
                )
            else:
                print(
                    "PASS: No patient appears in multiple splits"
                )

            print("\nSplit distribution:")
            print(
                splits[split_column]
                .value_counts()
                .to_string()
            )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    print_heading("VALIDATION RESULT")

    if warnings:
        print("\nWarnings:")

        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nErrors:")

        for error in errors:
            print(f"  - {error}")

        print(
            f"\nVALIDATION FAILED: "
            f"{len(errors)} error(s), "
            f"{len(warnings)} warning(s)"
        )

        return False

    print(
        f"\nVALIDATION PASSED: "
        f"0 errors, {len(warnings)} warning(s)"
    )

    return True


if __name__ == "__main__":
    validation_passed = validate_dataset()

    if validation_passed:
        sys.exit(0)

    sys.exit(1)