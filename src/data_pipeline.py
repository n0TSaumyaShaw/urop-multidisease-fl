import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_ehr_pipeline(
    raw_dir="../data/raw",
    output_path="../data/processed/preprocessed_ehr_dataset.csv"
):
    print("--- Step 1: Cohort Selection & Filtering ---")
    patients = pd.read_csv(os.path.join(raw_dir, "patients.csv"))
    admissions = pd.read_csv(os.path.join(raw_dir, "admissions.csv"))
    
    cohort = pd.merge(admissions, patients, on="subject_id", how="inner")
    cohort['admittime'] = pd.to_datetime(cohort['admittime'], errors='coerce')
    cohort['dischtime'] = pd.to_datetime(cohort['dischtime'], errors='coerce')
    cohort = cohort.dropna(subset=['admittime', 'dischtime']).copy()
    
    cohort['admission_year'] = cohort['admittime'].dt.year
    cohort['age'] = cohort['anchor_age'] + (cohort['admission_year'] - cohort['anchor_year'])
    cohort = cohort[cohort['age'] >= 18].copy()
    print(f"Total valid adult admissions: {len(cohort)}")
    
    print("\n--- Step 2: Clinical Grouping & Target Encoding ---")
    diagnoses = pd.read_csv(os.path.join(raw_dir, "diagnoses_icd.csv"))
    
    disease_map = {
        '4019': 'Hypertension', 'I10': 'Hypertension',
        '2724': 'Hyperlipidemia', 'E785': 'Hyperlipidemia',
        '25000': 'Diabetes', 'E119': 'Diabetes',
        '42731': 'Atrial_Fibrillation', 'I4891': 'Atrial_Fibrillation'
    }
    diagnoses['disease'] = diagnoses['icd_code'].map(disease_map)
    mapped_dx = diagnoses.dropna(subset=['disease']).copy()
    
    disease_matrix = (pd.crosstab(mapped_dx['hadm_id'], mapped_dx['disease']) > 0).astype(int).reset_index()
    
    target_cols = ['Hypertension', 'Hyperlipidemia', 'Diabetes', 'Atrial_Fibrillation']
    cohort = pd.merge(cohort[['subject_id', 'hadm_id', 'gender', 'age']], disease_matrix, on='hadm_id', how='left')
    cohort[target_cols] = cohort[target_cols].fillna(0).astype(int)
    cohort['gender_encoded'] = (cohort['gender'] == 'M').astype(int)
    
    print(f"Target distribution:\n{cohort[target_cols].sum()}")
    
    print("\n--- Step 3: Feature Extraction (Labs & Vitals) ---")
    labs = pd.read_csv(os.path.join(raw_dir, "labevents.csv"))
    charts = pd.read_csv(os.path.join(raw_dir, "chartevents.csv"))
    
    lab_map = {50971: 'Potassium', 50983: 'Sodium'}
    labs_filt = labs[labs['itemid'].isin(lab_map.keys())].copy()
    labs_filt['feature'] = labs_filt['itemid'].map(lab_map)
    lab_summary = labs_filt.groupby(['hadm_id', 'feature'])['valuenum'].mean().unstack().reset_index()
    
    chart_map = {220045: 'Heart_Rate', 220210: 'Respiratory_Rate'}
    charts_filt = charts[charts['itemid'].isin(chart_map.keys())].copy()
    charts_filt['feature'] = charts_filt['itemid'].map(chart_map)
    chart_summary = charts_filt.groupby(['hadm_id', 'feature'])['valuenum'].mean().unstack().reset_index()
    
    cohort = pd.merge(cohort, lab_summary, on='hadm_id', how='left')
    cohort = pd.merge(cohort, chart_summary, on='hadm_id', how='left')
    
    print("\n--- Step 4: Data Cleaning & Imputation ---")
    num_features = ['age', 'Potassium', 'Sodium', 'Heart_Rate', 'Respiratory_Rate']
    cohort[num_features] = cohort[num_features].fillna(cohort[num_features].mean())
    
    print("\n--- Step 5: Feature Normalization ---")
    scaler = MinMaxScaler()
    scaled_feature_cols = [f"{col}_scaled" for col in num_features]
    cohort[scaled_feature_cols] = scaler.fit_transform(cohort[num_features])
    
    final_feature_cols = ['gender_encoded'] + scaled_feature_cols
    final_output = cohort[['subject_id', 'hadm_id'] + final_feature_cols + target_cols]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_output.to_csv(output_path, index=False)
    print(f"\nSaved preprocessed dataset to: {output_path}")
    print(f"Shape: {final_output.shape}")
    return final_output

if __name__ == "__main__":
    preprocess_ehr_pipeline(raw_dir="data/raw", output_path="data/processed/preprocessed_ehr_dataset.csv")