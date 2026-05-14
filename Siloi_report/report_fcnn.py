import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# Function to train the model
def train_model(model, train_loader, criterion=nn.MSELoss(), optimizer=None, num_epochs=100):
    """
    Trains the given model using the provided training data loader, loss criterion, and optimizer.
    Arguments:
        - model: The neural network model to be trained.
        - train_loader: DataLoader providing the training data.
        - criterion: Loss function to optimize (default is Mean Squared Error).
        - optimizer: Optimization algorithm to update model parameters (e.g., Adam, SGD).
        - num_epochs: Number of epochs to train the model (default is 100).
    Returns:
        - None (the model is trained in-place).
    """
    model.train()
    for epoch in range(num_epochs):
        for s, l in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(s), l)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Function to evaluate the model
def evaluate_model(model, test_loader):
    """
    Evaluates the trained model on the test dataset and calculates the Mean Absolute Error (MAE).
    Arguments:
        - model: The trained neural network model to be evaluated.
        - test_loader: DataLoader providing the test data.
    Returns:
        - mae: The Mean Absolute Error of the model's predictions on the test dataset.
    """
    model.eval()
    abs_err = 0
    with torch.no_grad():
        for s, l in test_loader:
            preds = model(s)
            abs_err += torch.abs(preds - l).sum().item()
    
    mae = abs_err / len(test_loader.dataset)
    return mae


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

# --- STEP 3: DEEP REGRESSION NETWORK ---
class OptimizedGPANet(nn.Module):
    def __init__(self, input_size, activation=nn.ReLU, dropout_rate=0.2):
        super(OptimizedGPANet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

# --- STEP 4: MAIN EXECUTION ---
if __name__ == "__main__":

    # Dataset and feature/target setup
    path = r'Teen_Mental_Health_Dataset.csv'    
    features = ["age", "gender", "daily_social_media_hours", "sleep_hours", 
                "screen_time_before_sleep", "physical_activity", 
                "stress_level", "anxiety_level", "addiction_level"]
    target = "academic_performance"
    dataset = GPADataset(path, features, target)

    # Defining the sizes of the training and testing sets (based on the length of the dataset)
    train_set_size = 0.7
    test_set_size = 1 - train_set_size
    train_set, test_set = random_split(dataset, [int(train_set_size*len(dataset)), int(test_set_size*len(dataset))],
                                       generator=torch.Generator().manual_seed(42))
    

    # Hyperparameters and model setup
    lr, num_epochs, dropout_rate = 0.01, 100, 0.2
    model = OptimizedGPANet(len(features), dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DataLoaders for training and testing
    train_batch_size = 32
    test_batch_size = 32
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)


    print("--- Final Model: Estimating Impact on Academic Performance ---")

    # Train the model
    train_model(model, train_loader, optimizer=optimizer, num_epochs=num_epochs)

    # Evaluation
    mae = evaluate_model(model, test_loader)    
    avg_gpa = pd.read_csv(path)[target].mean()
    accuracy = (1 - (mae / avg_gpa)) * 100

    print(f"\n--- FINAL RESULTS ---")
    print(f"Target: {target}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Relative Accuracy: {accuracy:.2f} %")