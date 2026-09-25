# Synthea Disease and Schema Profile

## Dataset Population
- Total patient records: 6213
- Living patients: 5000
- Deceased patients: 1213

## Available Columns
### patients.csv
Id,BIRTHDATE,DEATHDATE,SSN,DRIVERS,PASSPORT,PREFIX,FIRST,MIDDLE,LAST,SUFFIX,MAIDEN,MARITAL,RACE,ETHNICITY,GENDER,BIRTHPLACE,ADDRESS,CITY,STATE,COUNTY,FIPS,ZIP,LAT,LON,HEALTHCARE_EXPENSES,HEALTHCARE_COVERAGE,INCOME
### conditions.csv
START,STOP,PATIENT,ENCOUNTER,SYSTEM,CODE,DESCRIPTION
### encounters.csv
Id,START,STOP,PATIENT,ORGANIZATION,PROVIDER,PAYER,ENCOUNTERCLASS,CODE,DESCRIPTION,BASE_ENCOUNTER_COST,TOTAL_CLAIM_COST,PAYER_COVERAGE,REASONCODE,REASONDESCRIPTION
### observations.csv
DATE,PATIENT,ENCOUNTER,CATEGORY,CODE,DESCRIPTION,VALUE,UNITS,TYPE

## Target Disease Availability
| Research label | Synthea description | Code | Records | Unique patients |
|---|---|---|---:|---:|
| Hypertension | Essential hypertension (disorder) | 59621000 | 1953 | 1953 |
| Hyperlipidemia | Hyperlipidemia (disorder) | 55822004 | 1028 | 1953 |
| Diabetes | Diabetes mellitus type 2 (disorder) | 44054006 | 694  | 694  |
| Atrial fibrillation | Atrial fibrillation (disorder) | 49436004 | 89 | 89 |

## Observations
- Several Synthea descriptions may map to one research label.
- Final label mapping requires review before synthea_pipeline.py.
- Synthea and MIMIC have not been combined.
