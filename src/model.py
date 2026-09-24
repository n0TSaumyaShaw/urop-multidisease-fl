import torch
import torch.nn as nn

class MultiDiseaseNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=4):
        """
        Baseline Neural Network for Multi-Disease Risk Prediction.
        input_dim: 6 clinical features
        output_dim: 4 independent disease probabilities
        """
        super(MultiDiseaseNN, self).__init__()
        
        # Fully connected feed-forward network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Prevents overfitting on small datasets
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim) 
            # Notice there is NO Sigmoid at the end. 
            # BCEWithLogitsLoss handles the Sigmoid activation automatically.
        )

    def forward(self, x):
        """
        Forward pass of the neural network.
        Outputs 'logits' (raw, unnormalized predictions).
        """
        logits = self.network(x)
        return logits

# Quick test to ensure the model compiles
if __name__ == "__main__":
    # Simulate one patient's 6 clinical features
    dummy_patient = torch.randn(1, 6) 
    
    model = MultiDiseaseNN()
    raw_predictions = model(dummy_patient)
    
    print("Model initialized successfully!")
    print(f"Input shape: {dummy_patient.shape} -> Output shape: {raw_predictions.shape}")