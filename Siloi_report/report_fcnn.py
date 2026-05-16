# "Data handling and visualization" libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# "Modeling" libraries
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.special

# "Evaluation and hyperparameter optimization" libraries
import optuna
from sklearn.metrics import classification_report, confusion_matrix


# Global paths to dataset and model storage
DATA_PATH = 'data.csv'
MODEL_DB_PATH = 'cancer_study.db'

# Loss function for binary classification (logistic regression)
CRITERION = nn.BCEWithLogitsLoss()

# --- STEP 1: PREPROCESSING ---
def data_load_clean(data_file):
    data = pd.read_csv(data_file) 
    # Map categories to numbers
    # Benign (B) = 0, Malignant (M) = 1
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1.0})
    
    return data

# --- STEP 2: REGRESSION DATASET ---
class CancerDataset(Dataset):
    def __init__(self, X, y):
        self.samples = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def __len__(self): return len(self.labels)
    def __getitem__(self, index): return self.samples[index], self.labels[index]


# --- STEP 3.1: DEEP REGRESSION NETWORK - PYTORCH ---
class OptimizedGPANet(nn.Module):
    def __init__(self, input_size, p_drop):         
        super(OptimizedGPANet, self).__init__()
        self.net = nn.Sequential(
            # First layer
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            # Second layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p_drop),
            # Third layer
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p_drop),
            # Output layer
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
    
    # Parameters to optimize
    drop = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)

    num_feat = len(features)
    model = OptimizedGPANet(input_size=num_feat, p_drop=drop)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_model(model, num_epochs=150, train_loader=train_loader, test_loader=test_loader, 
                optimizer=optimizer, criterion=CRITERION, verbose=False)

    test_loss = 0.0
    model.eval() 
    
    with torch.no_grad():
        for s, l in test_loader:
            batch_loss = CRITERION(model(s), l)
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

    study.optimize(objective, n_trials=40)
    return study.best_params


# --- STEP 4: MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Load and preprocess the data
    df = data_load_clean(DATA_PATH)

    # Define features and target
    features = df.columns[2:-1].tolist()
    target = 'diagnosis'

    X = df[features].values
    y = df[target].values

    # Train-test split and scaling
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and load datasets
    train_set = CancerDataset(X_train, y_train)
    test_set = CancerDataset(X_test, y_test)
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

    # Model training
    train_model(model=model, num_epochs=250, train_loader=train_loader, 
                test_loader=test_loader, optimizer=optimizer, criterion=CRITERION, patience=20, verbose=True)

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
    threshold = 0.5
    probs = scipy.special.expit(all_preds_np)  # Convert logits to probabilities
    metrics = classification_report(all_labels_np, (probs > threshold).astype(int), output_dict=True)

    # Results presentation
    print(f"Classification report: {metrics}\n")


    # Feature correlation heatmap
    df_clean = data_load_clean(DATA_PATH)
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_clean[features + [target]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    
    plt.title("Correlation Heatmap of Features and Target")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("Correlation heatmap saved as 'correlation_heatmap.png'.")