import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import the brain we just built
from model import MultiDiseaseNN

# 1. Define how PyTorch reads our CSV
class EHRDataset(Dataset):
    def __init__(self, csv_file):
        # Load the preprocessed data
        self.data = pd.read_csv(csv_file)
        
        # Separate the 6 inputs (Features)
        feature_cols = ['gender_encoded', 'age_scaled', 'Potassium_scaled', 
                        'Sodium_scaled', 'Heart_Rate_scaled', 'Respiratory_Rate_scaled']
        self.features = self.data[feature_cols].values
        
        # Separate the 4 answers (Targets)
        target_cols = ['Hypertension', 'Hyperlipidemia', 'Diabetes', 'Atrial_Fibrillation']
        self.targets = self.data[target_cols].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert pandas rows into PyTorch math tensors
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

def train_model():
    print("--- Step 1: Loading Dataset ---")
    # Point it to the CSV we generated in Objective 1
    dataset = EHRDataset('data/processed/preprocessed_ehr_dataset.csv')
    
    # DataLoader feeds the data to the model in randomized batches of 16 patients
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"Loaded {len(dataset)} patients.")

    print("\n--- Step 2: Initializing Model & Optimizers ---")
    model = MultiDiseaseNN()
    
    # BCEWithLogitsLoss: The required math function for Multi-Label learning
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam Optimizer: The algorithm that actually updates the weights to reduce errors
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 50
    print(f"\n--- Step 3: Starting Training Loop ({epochs} Epochs) ---")
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Loop through the data in batches of 16
        for batch_x, batch_y in dataloader:
            
            # A. Clear the old memory
            optimizer.zero_grad()
            
            # B. Forward Pass (Make a guess)
            predictions = model(batch_x)
            
            # C. Calculate the Error (Loss)
            loss = criterion(predictions, batch_y)
            
            # D. Backward Pass (Learn from the mistake)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Error (Loss): {avg_loss:.4f}")
            
    print("\nTraining complete! The baseline AI is successfully learning.")
    
    # Save the trained model's "brain state"
    torch.save(model.state_dict(), "data/processed/baseline_model.pth")
    print("Saved model weights to data/processed/baseline_model.pth")

if __name__ == "__main__":
    train_model()