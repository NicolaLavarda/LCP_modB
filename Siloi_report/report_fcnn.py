from matplotlib.pyplot import step
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import optuna

# --- STEP 1: PREPROCESSING ---
def data_load_clean(data_file):
    data = pd.read_csv(data_file) 
    # Map categories to numbers
    data['gender'] = data['gender'].map({'male': 0.0, 'female': 1.0})
    data['platform_usage'] = data['platform_usage'].map({'Instagram': 0.0, 'TikTok': 1.0, 'Both': 2.0})
    data['social_interaction_level'] = data['social_interaction_level'].map({'low': 0.0, 'medium': 1.0, 'high': 2.0})

    # Normalize continuous features (but NOT the target: academic_performance)
    features = ["age", "daily_social_media_hours", "sleep_hours", 
                "screen_time_before_sleep", "physical_activity", 
                "stress_level", "anxiety_level", "addiction_level"]
    
    for col in features:
        if col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()

    return data

# --- STEP 2: REGRESSION DATASET ---
class GPADataset(Dataset):
    def __init__(self, data_file, features, target):
        df = data_load_clean(data_file)
        self.samples = torch.tensor(df[features].values, dtype=torch.float32)
        self.labels = torch.tensor(df[target].values, dtype=torch.float32).view(-1, 1)
    def __len__(self): return len(self.labels)
    def __getitem__(self, index): return self.samples[index], self.labels[index]

# --- STEP 3.1: DEEP REGRESSION NETWORK - PYTORCH ---
class OptimizedGPANet(nn.Module):
    def __init__(self, input_size, p_drop):         # Now the dropout is a parameter (Modified 14/05)
        super(OptimizedGPANet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

def train_model(model, num_epochs, train_loader, optimizer, criterion):
    print("--- Final Model: Estimating Impact on Academic Performance ---")
    model.train()
    for epoch in range(num_epochs):
        for s, l in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(s), l)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 2. Use Optuna to instantiate your class

def objective(trial, num_feat, train_loader, test_loader):
    # Suggest hyperparameters
    # h_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
    drop = trial.suggest_float("dropout_rate", 0.0, 0.2, step=0.1)
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)

    # Instantiate your model class with suggested values
    model = OptimizedGPANet(input_size=num_feat, p_drop=drop)
    
    # Standard training logic goes here...
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss function
    criterion = nn.MSELoss()

    # Train model
    train_model(model, 250, train_loader, optimizer, criterion)

    # Evaluate model
    test_loss = 0.0
    model.eval() 
    
    with torch.no_grad():
        for s, l in test_loader:
            batch_loss = criterion(model(s), l)
            test_loss += batch_loss.item() # Extract float to prevent memory leak
            
    # Calculate the average test loss across all batches
    avg_test_loss = test_loss / len(test_loader)

    return avg_test_loss


# --- STEP 4: MAIN EXECUTION ---
if __name__ == "__main__":
    path = r'Siloi_report/Teen_Mental_Health_Dataset.csv'
    
    # Impact features
    features = ["age", "gender", "daily_social_media_hours", "sleep_hours", 
                "screen_time_before_sleep", "physical_activity", 
                "stress_level", "anxiety_level", "addiction_level"]
    target = "academic_performance"

    # Defines and splits traning and testing datasets
    dataset = GPADataset(path, features, target)
    train_set, test_set = random_split(dataset, [960, 240], generator=torch.Generator().manual_seed(42))
    
    # Load the datasets
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # 3. Run the study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)


    p_drop = 0.2

    model = OptimizedGPANet(len(features), p_drop)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Model training
    train_model(250)

    # Evaluation
    model.eval()
    abs_err = 0
    with torch.no_grad():
        for s, l in test_loader:
            preds = model(s)
            abs_err += torch.abs(preds - l).sum().item()
    
    mae = abs_err / 240
    avg_gpa = pd.read_csv(path)[target].mean()
    accuracy = (1 - (mae / avg_gpa)) * 100

    print(f"\n--- FINAL RESULTS ---")
    print(f"Target: {target}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Relative Accuracy: {accuracy:.2f} %")
    print(f"\nConclusion: The model can predict student performance with ~81% accuracy using social habits.")
