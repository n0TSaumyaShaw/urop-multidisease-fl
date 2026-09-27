# Synthea Disease and Clinical Feature Profile

## 1. Purpose

This document records the disease targets and clinical features available in the generated Synthea dataset.

The dataset will be used to develop and evaluate the larger version of the federated multi-disease risk-prediction system. It must not be directly appended to the existing MIMIC-derived dataset because the two datasets have different structures, patient identifiers, clinical coding systems and feature definitions.

---

## 2. Dataset Summary

- Dataset source: Synthea synthetic patient generator
- Dataset type: Synthetic Electronic Health Records
- Requested population: 5,000 patients
- Exported patient records: 6,213
- Random seed: 42
- Requested age range: 18–90 years
- Export format: CSV
- FHIR export: Disabled
- Generation date: 25 September 2026

The exported population contains 5,000 living patients and 1,213 deceased patients generated as part of their longitudinal health histories.

---

## 3. Available Raw Files

The following files are available under:

```text
data/raw/synthea/
```

### 3.1 `patients.csv`

Contains patient demographic information.

Important columns include:

- `Id`
- `BIRTHDATE`
- `DEATHDATE`
- `MARITAL`
- `RACE`
- `ETHNICITY`
- `GENDER`
- `BIRTHPLACE`
- `CITY`
- `STATE`
- `COUNTY`
- `ZIP`
- `HEALTHCARE_EXPENSES`
- `HEALTHCARE_COVERAGE`
- `INCOME`

The `Id` column is the unique patient identifier.

### 3.2 `conditions.csv`

Contains diagnosed patient conditions and diseases.

Important columns include:

- `START`
- `STOP`
- `PATIENT`
- `ENCOUNTER`
- `SYSTEM`
- `CODE`
- `DESCRIPTION`

The `PATIENT` column connects each condition to a patient in `patients.csv`.

### 3.3 `encounters.csv`

Contains healthcare encounters such as ambulatory, outpatient, emergency and inpatient visits.

Important columns include:

- `Id`
- `START`
- `STOP`
- `PATIENT`
- `ORGANIZATION`
- `PROVIDER`
- `PAYER`
- `ENCOUNTERCLASS`
- `CODE`
- `DESCRIPTION`
- `BASE_ENCOUNTER_COST`
- `TOTAL_CLAIM_COST`
- `PAYER_COVERAGE`
- `REASONCODE`
- `REASONDESCRIPTION`

The `Id` column identifies the encounter, while `PATIENT` connects the encounter to a patient.

### 3.4 `observations.csv`

Contains laboratory measurements, vital signs and other clinical observations.

Important columns include:

- `DATE`
- `PATIENT`
- `ENCOUNTER`
- `CATEGORY`
- `CODE`
- `DESCRIPTION`
- `VALUE`
- `UNITS`
- `TYPE`

The `PATIENT` column connects observations to patients, and `ENCOUNTER` connects observations to healthcare encounters.

---

## 4. Candidate Disease Targets

Disease counts were calculated after reading the `CODE` column as text to preserve SNOMED CT codes correctly.

| Disease | SNOMED CT Code | Condition Records | Unique Patients |
|---|---:|---:|---:|
| Hypertension | 59621000 | 1,953 | 1,953 |
| Hyperlipidemia | 55822004 | 1,028 | 1,028 |
| Type 2 Diabetes | 44054006 | 694 | 694 |
| Atrial Fibrillation | 49436004 | 89 | 89 |
| Ischemic Heart Disease | 414545008 | 1,833 | 1,833 |

Each condition currently has one matching record per affected patient for the selected codes.

---

## 5. Disease-Target Decision

The proposed primary disease targets for the larger Synthea experiment are:

1. Hypertension
2. Hyperlipidemia
3. Type 2 Diabetes
4. Ischemic Heart Disease

These four diseases have enough positive patients to support training, validation, testing and federated-client simulation.

Atrial fibrillation will be retained as an optional rare-disease experiment. It is not recommended as one of the four primary targets because only 89 patients have the condition. This limited number could produce unstable evaluation results, especially after dividing patients into training, validation, test and federated-client groups.

The final target decision should be confirmed before implementing the Synthea preprocessing pipeline.

---

## 6. Clinical Feature Availability

The following potentially useful clinical features were identified in `observations.csv`.

| Clinical Feature | Units | Records | Unique Patients |
|---|---|---:|---:|
| Respiratory rate | `/min` | 95,225 | 6,213 |
| Systolic blood pressure | `mm[Hg]` | 104,319 | 6,213 |
| Diastolic blood pressure | `mm[Hg]` | 104,319 | 6,213 |
| Heart rate | `/min` | 95,225 | 6,213 |
| Body mass index | `kg/m2` | 92,581 | 6,195 |
| LDL cholesterol | `mg/dL` | 49,604 | 5,149 |
| Total cholesterol | `mg/dL` | 49,604 | 5,149 |
| HDL cholesterol | `mg/dL` | 49,604 | 5,149 |
| Sodium in blood | `mmol/L` | 86,402 | 3,749 |
| Potassium in blood | `mmol/L` | 86,402 | 3,749 |
| Glucose in blood | `mg/dL` | 86,402 | 3,749 |
| HbA1c | `%` | 76,240 | 3,524 |
| Sodium in serum or plasma | `mmol/L` | 58,275 | 1,663 |
| Potassium in serum or plasma | `mmol/L` | 58,275 | 1,663 |
| Glucose in serum or plasma | `mg/dL` | 58,275 | 1,663 |
| Glucose by test strip | `mg/dL` | 67,191 | 1,582 |
| BMI percentile for age and sex | `%` | 6,158 | 928 |
| Glucose presence in urine | `{nominal}` | 66,281 | 942 |
| Glucose presence in urine by test strip | `{presence}` | 1,383 | 515 |

---

## 7. Recommended Model Features

The first version of the Synthea model should use features with strong patient coverage and clear clinical meaning.

### Demographic features

- Age
- Gender
- Race
- Ethnicity
- Marital status
- Income

### Vital-sign features

- Systolic blood pressure
- Diastolic blood pressure
- Heart rate
- Respiratory rate
- Body mass index

### Laboratory features

- Blood glucose
- HbA1c
- Total cholesterol
- LDL cholesterol
- HDL cholesterol
- Sodium
- Potassium

### Healthcare-utilization features

- Total number of previous encounters
- Number of inpatient encounters
- Number of outpatient encounters
- Number of emergency encounters
- Time since the most recent encounter

Not every proposed feature must be included in the first implementation. The initial Synthea pipeline should begin with reliable demographic, vital-sign and laboratory features before adding more complex longitudinal features.

---

## 8. Data-Processing Requirements

The Synthea preprocessing pipeline should:

1. Read the four raw CSV files independently.
2. Preserve patient and encounter identifiers as strings.
3. Parse all date columns as dates.
4. calculate patient age using the prediction or index date.
5. Map SNOMED CT disease codes to multi-label binary targets.
6. Convert observation values to numeric values where applicable.
7. Standardize observations that describe the same clinical measurement.
8. Handle repeated observations using a defined summary method, such as the most recent value before the prediction date.
9. Prevent observations recorded after a disease diagnosis from being used to predict that diagnosis.
10. Create missingness indicators for clinical features.
11. Keep missing values unscaled in the raw processed output.
12. Split patients, rather than individual rows, into training, validation and test groups.
13. Fit imputers and scalers using only the training group.
14. Store preprocessing objects so the same transformations can be applied during federated training and real-time prediction.

---

## 9. Important Leakage Warning

Using measurements recorded after a disease diagnosis would create data leakage. The model might appear highly accurate because it is using information generated after the disease was already identified.

For early-risk prediction, each patient must have an index date. Only demographic information, encounters, vital signs and laboratory measurements recorded before that index date should be used as model inputs.

Disease labels should represent diagnoses occurring during a clearly defined future prediction window.

The initial Synthea implementation may first build a simpler patient-level prevalence baseline. However, this must be clearly described as disease classification rather than true early-disease prediction. A temporal early-risk cohort should then be developed for the final research experiments.

---

## 10. Dataset Suitability

The Synthea dataset is suitable for strengthening Objectives 1–4 because:

- It contains more than 6,000 synthetic patients.
- It does not contain real patient identities.
- It contains longitudinal clinical histories.
- Vital signs are available for nearly all patients.
- Important laboratory measurements are available for thousands of patients.
- The selected primary diseases have substantially more positive cases than the current demonstration dataset.
- Patients can be divided among multiple simulated healthcare institutions.
- Different client distributions can be created for federated-learning experiments.

The dataset is suitable for system development and experimental simulation. However, because it is synthetic, successful results should not be interpreted as evidence that the model is clinically validated or ready for use with real patients.

---

## 11. Next Development Step

After confirming this profile, the next task is to create:

```text
src/synthea_pipeline.py
```

This pipeline will convert the raw Synthea files into a consistent patient-level dataset for centralized and federated model training.

The raw Synthea files must remain separate from the existing MIMIC-derived files. They should not be manually appended to:

```text
data/processed/preprocessed_ehr_dataset.csv
```

The new Synthea pipeline should produce its own processed output, preprocessing metadata and validation report.