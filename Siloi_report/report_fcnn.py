import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import optuna
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Global paths to dataset and model storage
DATA_PATH = 'Teen_Mental_Health_Dataset.csv'
MODEL_DB_PATH = "GPANet_study.db"


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
    def __init__(self, input_size, p_drop):         
        super(OptimizedGPANet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

# Function to train the model with early stopping
def train_model(model, num_epochs, train_loader, test_loader, optimizer, criterion, patience=10, verbose=True):
    if verbose:
        print(f"--- Training started (Max Epochs: {num_epochs}, Patience: {patience}) ---")
        
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training the model
        model.train()
        for s, l in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(s), l)
            loss.backward()
            optimizer.step()

        # Validate the model on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s, l in test_loader:
                val_loss += criterion(model(s), l).item()
        avg_val_loss = val_loss / len(test_loader)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            # Model is improving, save the best loss and reset patience counter
            best_val_loss = avg_val_loss
            patience_counter = 0  
        else:
            patience_counter += 1 # Model is not improving, increment patience counter

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}")

        # Patience check to stop training if no improvement
        if patience_counter >= patience:
            if verbose:
                print(f"Triggered early stopping at epoch {epoch+1}!")
            break
            
    return best_val_loss



# Function to be optimized using optuna
def objective(trial):
    drop = trial.suggest_float("dropout_rate", 0.0, 0.2, step=0.1)
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)

    num_feat = 9
    model = OptimizedGPANet(input_size=num_feat, p_drop=drop)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_model(model, num_epochs=150, train_loader=train_loader, test_loader=test_loader, 
                optimizer=optimizer, criterion=criterion, verbose=False)

    test_loss = 0.0
    model.eval() 
    
    with torch.no_grad():
        for s, l in test_loader:
            batch_loss = criterion(model(s), l)
            test_loss += batch_loss.item() 
            
    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss

def find_best_par(objective, path=MODEL_DB_PATH):
    storage_name = f"sqlite:///{path}"
    
    if os.path.isfile(path):
        print("removing previous study db...")
        os.remove(path)
    else:
        print("creating new study db...")

    study = optuna.create_study(
        direction="minimize", 
        study_name="gpa_model_tuning", 
        storage=storage_name,
        load_if_exists=True 
    )

    study.optimize(objective, n_trials=20)
    return study.best_params


# --- STEP 4: MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Impact features
    features = ["age", "gender", "daily_social_media_hours", "sleep_hours", 
                "screen_time_before_sleep", "physical_activity", 
                "stress_level", "anxiety_level", "addiction_level"]
    target = "academic_performance"

    # Defines and splits traning and testing datasets
    dataset = GPADataset(DATA_PATH, features, target)
    train_set, test_set = random_split(dataset, [960, 240], generator=torch.Generator().manual_seed(42))
    
    # Load the datasets
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Best hyperparameters search using optuna (user choice)
    opt = input("Do you want to perform hyperparameter tuning with optuna? (y/n): ").strip().lower()
    if opt == 'y':
        best_parameters = find_best_par(objective, path=MODEL_DB_PATH)
        print(f"\nBest hyperparameters found: {best_parameters}\n")
    else:
        db = optuna.storages.RDBStorage(url=f"sqlite:///{MODEL_DB_PATH}")
        study = optuna.load_study(study_name="gpa_model_tuning", storage=db)
        best_parameters = study.best_params
        print(f"\nBest hyperparameters loaded from previous study: {best_parameters}\n")

    # Sets the definitive parameters 
    p_drop, learning_rate = best_parameters['dropout_rate'], best_parameters['lr'] 

    # Defines the definitive model
    model = OptimizedGPANet(len(features), p_drop)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Model training
    train_model(model=model, num_epochs=250, train_loader=train_loader, 
                test_loader=test_loader, optimizer=optimizer, criterion=criterion)

    # Model evaluation
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for s, l in test_loader:
            preds = model(s)
            all_preds.append(preds.cpu())
            all_labels.append(l.cpu())

    all_preds_np = torch.cat(all_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy()

    # Metrics calculation
    mae = mean_absolute_error(all_labels_np, all_preds_np)
    mse = mean_squared_error(all_labels_np, all_preds_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels_np, all_preds_np)

    avg_gpa = pd.read_csv(DATA_PATH)[target].mean()
    accuracy = (1 - (mae / avg_gpa)) * 100


    # Results presentation
    print(f"\n--- FINAL RESULTS ---")
    print(f"Target: {target}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Relative Accuracy: {accuracy:.2f} %")


    # Scatter plot of real vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels_np, all_preds_np, alpha=0.5, color='blue', label='Model Predictions')
    min_val = min(all_labels_np.min(), all_preds_np.min())
    max_val = max(all_labels_np.max(), all_preds_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
    
    plt.title('Academic Performances: Real vs Predicted')
    plt.xlabel('GPA Real')
    plt.ylabel('GPA Predicted')
    plt.legend()
    plt.grid(True)
    plt.savefig('scatter_plot.png')
    print("Scatter plot saved as 'scatter_plot.png'.")


    # Feature correlation heatmap
    df_clean = data_load_clean(DATA_PATH)
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_clean[features + [target]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    
    plt.title("Correlation Heatmap of Features and Target")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("Correlation heatmap saved as 'correlation_heatmap.png'.")