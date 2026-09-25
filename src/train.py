import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import copy

from model import MultiDiseaseNN

# 1. Lock random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EHRDataset(Dataset):
    def __init__(self, data_frame, feature_cols, target_cols):
        self.features = data_frame[feature_cols].values
        self.targets = data_frame[target_cols].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

def evaluate_model(model, dataloader, criterion, target_cols):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_preds.append(probs.numpy())
            all_targets.append(batch_y.numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    binary_preds = (all_preds > 0.5).astype(int)
    
    metrics = {}
    for i, disease in enumerate(target_cols):
        # Handle edge cases where a small test set might only have 1 class for a specific disease
        valid_roc = len(np.unique(all_targets[:, i])) > 1
        
        metrics[disease] = {
            'ROC-AUC': roc_auc_score(all_targets[:, i], all_preds[:, i]) if valid_roc else float('nan'),
            'Precision': precision_score(all_targets[:, i], binary_preds[:, i], zero_division=0),
            'Recall': recall_score(all_targets[:, i], binary_preds[:, i], zero_division=0),
            'F1-Score': f1_score(all_targets[:, i], binary_preds[:, i], zero_division=0)
        }
    return total_loss / len(dataloader), metrics

def train_model():
    set_seed(42)
    print("--- Step 1: Loading & Patient-Level Splitting ---")
    data = pd.read_csv('data/processed/preprocessed_ehr_dataset.csv')
    
    feature_cols = ['gender_encoded', 'age_scaled', 'Potassium_scaled', 
                    'Sodium_scaled', 'Heart_Rate_scaled', 'Respiratory_Rate_scaled']
    target_cols = ['Hypertension', 'Hyperlipidemia', 'Diabetes', 'Atrial_Fibrillation']
    
    # Isolate unique patients to prevent data leakage
    unique_subjects = data['subject_id'].unique()
    
    # 70% Train, 15% Validation, 15% Test
    train_subj, temp_subj = train_test_split(unique_subjects, test_size=0.30, random_state=42)
    val_subj, test_subj = train_test_split(temp_subj, test_size=0.50, random_state=42)
    
    train_data = data[data['subject_id'].isin(train_subj)]
    val_data = data[data['subject_id'].isin(val_subj)]
    test_data = data[data['subject_id'].isin(test_subj)]
    
    train_dataset = EHRDataset(train_data, feature_cols, target_cols)
    val_dataset = EHRDataset(val_data, feature_cols, target_cols)
    test_dataset = EHRDataset(test_data, feature_cols, target_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Split Breakdown (Admissions): Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    print("\n--- Step 2: Initializing Model & Handling Class Imbalance ---")
    model = MultiDiseaseNN()
    
    # Calculate pos_weight for BCEWithLogitsLoss: (Total Negatives) / (Total Positives)
    y_train = train_data[target_cols].values
    num_positives = y_train.sum(axis=0)
    num_negatives = len(y_train) - num_positives
    pos_weight = torch.tensor(num_negatives / np.maximum(num_positives, 1), dtype=torch.float32)
    
    print(f"Applied Positive Weights for Imbalance: {pos_weight.numpy()}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 50
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    
    print("\n--- Step 3: Training Loop ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, _ = evaluate_model(model, val_loader, criterion, target_cols)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
    # Load best model for final testing
    model.load_state_dict(best_model_weights)
    torch.save(best_model_weights, "data/processed/baseline_model.pth")
    
    print("\n--- Step 4: Final Evaluation on Unseen Test Set ---")
    test_loss, test_metrics = evaluate_model(model, test_loader, criterion, target_cols)
    print(f"Test Set Loss: {test_loss:.4f}\n")
    
    for disease, metrics in test_metrics.items():
        print(f"--- {disease} ---")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    train_model()