# Synthea Dataset Information

## Source
- Dataset type: Synthetic electronic health records
- Generator: Synthea
- Generation method: Generated locally using the Synthea executable JAR
- Download source: https://github.com/synthetichealth/synthea/releases
- Synthea release: master-branch-latest
- JAR SHA-256: ADD JAR HASH
- Generation date: 25 September 2026

## Generation Parameters
- Requested living population: 5000
- Exported patient records: 6213
- Living patients: 5000
- Deceased patients: 1213
- Unique patient IDs: 6213
- Random seed: 42
- Age range: 18-90
- Location: Default Synthea location
- CSV export: Enabled
- FHIR export: Disabled

## Generation Command
java -jar synthea-with-dependencies.jar -p 5000 -s 42 -a 18-90 --exporter.csv.export=true --exporter.fhir.export=false

## Files Used
- patients.csv
- conditions.csv
- observations.csv
- encounters.csv

## Dataset Identifier
- patients.csv SHA-256: 4BEC7E4FAD87EC2CF316687E687C59D6B16DB983D2FBAD2064F4598CB034BD56

## Data Handling
Raw CSV files are stored under data/raw/synthea and excluded from Git.
The project copy is the canonical dataset. A later smaller run replaced
the generator output and must not overwrite the project copy.
