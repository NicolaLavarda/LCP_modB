import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from report_fcnn import GPADataset, OptimizedGPANet, data_load_clean

# --- ESTHETIC SETTINGS ---
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlepad'] = 15

def generate_analysis_plots(model, test_loader, features, df_norm, path_csv):
    model.eval()
    actuals = []
    predictions = []
    
    with torch.no_grad():
        for s, l in test_loader:
            preds = model(s)
            actuals.extend(l.view(-1).tolist())
            predictions.extend(preds.view(-1).tolist())
    
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    
    # 1. PARITY PLOT
    plt.figure()
    sns.regplot(x=actuals, y=predictions, scatter_kws={'alpha':0.4, 's':20}, line_kws={'color':'red'})
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], '--k', alpha=0.5)
    plt.title("Model Accuracy: Actual vs Predicted GPA", fontsize=14, fontweight='bold')
    plt.xlabel("Actual GPA", fontsize=12)
    plt.ylabel("Predicted GPA", fontsize=12)
    plt.tight_layout()
    plt.savefig("Siloi_report/plots/analysis_1_parity.png", dpi=300)
    
    # 2. RESIDUALS DISTRIBUTION
    plt.figure()
    residuals = actuals - predictions
    sns.histplot(residuals, kde=True, color="purple", bins=25)
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Error Distribution (Residuals)", fontsize=14, fontweight='bold')
    plt.xlabel("Prediction Error", fontsize=12)
    plt.tight_layout()
    plt.savefig("Siloi_report/plots/analysis_2_residuals.png", dpi=300)
    
    # 3. SOCIAL MEDIA SENSITIVITY (Model's Internal Logic)
    plt.figure()
    # Create synthetic profiles varying ONLY social media hours
    social_idx = features.index("daily_social_media_hours")
    mean_profile = torch.tensor(df_norm[features].mean().values, dtype=torch.float32).unsqueeze(0)
    
    hours_range = np.linspace(df_norm["daily_social_media_hours"].min(), 
                             df_norm["daily_social_media_hours"].max(), 100)
    
    sim_preds = []
    for h in hours_range:
        profile = mean_profile.clone()
        profile[0, social_idx] = h
        with torch.no_grad():
            sim_preds.append(model(profile).item())
            
    plt.plot(hours_range, sim_preds, color='orange', linewidth=3)
    plt.title("Model Sensitivity: Impact of Social Media on GPA", fontsize=14, fontweight='bold')
    plt.xlabel("Daily Social Media Hours (Normalized)", fontsize=12)
    plt.ylabel("Predicted GPA", fontsize=12)
    plt.tight_layout()
    plt.savefig("Siloi_report/plots/analysis_3_social_impact.png", dpi=300)

    # 4. SLEEP VS PERFORMANCE LOGIC
    plt.figure()
    sleep_idx = features.index("sleep_hours")
    sleep_range = np.linspace(df_norm["sleep_hours"].min(), df_norm["sleep_hours"].max(), 100)
    
    sim_preds_sleep = []
    for sl in sleep_range:
        profile = mean_profile.clone()
        profile[0, sleep_idx] = sl
        with torch.no_grad():
            sim_preds_sleep.append(model(profile).item())
            
    plt.plot(sleep_range, sim_preds_sleep, color='teal', linewidth=3)
    plt.title("Model Sensitivity: Impact of Sleep on GPA", fontsize=14, fontweight='bold')
    plt.xlabel("Sleep Hours (Normalized)", fontsize=12)
    plt.ylabel("Predicted GPA", fontsize=12)
    plt.tight_layout()
    plt.savefig("Siloi_report/plots/analysis_4_sleep_impact.png", dpi=300)

if __name__ == "__main__":
    # Setup
    csv_path = r'Siloi_report/Teen_Mental_Health_Dataset.csv'
    features = ["age", "gender", "daily_social_media_hours", "sleep_hours", 
                "screen_time_before_sleep", "physical_activity", 
                "stress_level", "anxiety_level", "addiction_level"]
    target = "academic_performance"
    
    # Load and Train briefly to have a working model for plotting
    dataset = GPADataset(csv_path, features, target)
    df_norm = data_load_clean(csv_path)
    
    train_set, test_set = random_split(dataset, [960, 240], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    model = OptimizedGPANet(len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    print("Quick training for analysis...")
    for epoch in range(150):
        for s, l in train_loader:
            optimizer.zero_grad()
            criterion(model(s), l).backward()
            optimizer.step()
            
    print("Generating analysis plots...")
    os.makedirs("Siloi_report/plots", exist_ok=True)
    generate_analysis_plots(model, test_loader, features, df_norm, csv_path)
    print("Analysis plots saved in 'Siloi_report/plots/'.")
