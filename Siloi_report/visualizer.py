import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Premium aesthetic settings
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlepad'] = 20
plt.rcParams['axes.labelpad'] = 15

def load_data(file_path):
    """Loads data while keeping categorical labels for plotting."""
    data = pd.read_csv(file_path)
    return data

def plot_age_stress_gender(df):
    """1. Age vs Stress colored by Gender (Pink/Blue palette)"""
    plt.figure(figsize=(12, 8))
    # Custom palette: blue for male, pink for female
    palette = {"male": "#5DADE2", "female": "#EC7063"}
    
    sns.stripplot(data=df, x="age", y="stress_level", hue="gender", 
                  palette=palette, alpha=0.6, jitter=0.3, size=8)
    
    plt.title("Impact of Age on Stress Level by Gender", fontsize=16, fontweight='bold')
    plt.xlabel("Age (Years)", fontsize=12)
    plt.ylabel("Stress Level (1-10)", fontsize=12)
    plt.legend(title="Gender", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("Siloi_report/plots/plot_age_stress_gender.png", dpi=300)
    #plt.show()

def plot_social_platform_stress(df):
    """2. Social Media Hours vs Platform with Stress Intensity (Colormap)"""
    plt.figure(figsize=(14, 8))
    
    scatter = plt.scatter(df['daily_social_media_hours'], df['platform_usage'], 
                         c=df['stress_level'], cmap='YlOrRd', s=100, alpha=0.7, edgecolors='w')
    
    plt.colorbar(scatter, label='Stress Level (Intensity)')
    plt.title("Social Usage and Platforms vs Stress Intensity", fontsize=16, fontweight='bold')
    plt.xlabel("Daily Social Media Hours", fontsize=12)
    plt.ylabel("Platform Usage", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Siloi_report/plots/plot_social_platform_stress.png", dpi=300)
    #plt.show()

def plot_sleep_performance_depression(df):
    """3. Sleep Hours vs Academic Performance colored by Depression"""
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(data=df, x="sleep_hours", y="academic_performance", 
                    hue="depression_label", size="screen_time_before_sleep",
                    sizes=(20, 200), palette="viridis", alpha=0.7)
    
    plt.title("Sleep, Academic Performance, and Depression Indicator", fontsize=16, fontweight='bold')
    plt.xlabel("Sleep Hours", fontsize=12)
    plt.ylabel("Academic Performance (GPA)", fontsize=12)
    plt.legend(title="Depression (0/1) and Screen Time", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("Siloi_report/plots/plot_sleep_performance_depression.png", dpi=300)
    #plt.show()

def plot_activity_anxiety_gender(df):
    """4. Physical Activity vs Anxiety colored by Gender"""
    plt.figure(figsize=(12, 8))
    palette = {"male": "#3498DB", "female": "#E74C3C"}
    
    sns.kdeplot(data=df, x="physical_activity", y="anxiety_level", 
                hue="gender", fill=True, palette=palette, alpha=0.4)
    
    plt.title("Density Distribution: Physical Activity vs Anxiety", fontsize=16, fontweight='bold')
    plt.xlabel("Physical Activity (Hours/Day)", fontsize=12)
    plt.ylabel("Anxiety Level", fontsize=12)
    plt.tight_layout()
    plt.savefig("Siloi_report/plots/plot_activity_anxiety_gender.png", dpi=300)
    #plt.show()

def plot_screentime_addiction_stress(df):
    """5. Screen Time Before Sleep vs Addiction with Stress Intensity"""
    plt.figure(figsize=(12, 8))
    
    points = plt.scatter(df['screen_time_before_sleep'], df['addiction_level'], 
                        c=df['stress_level'], s=df['daily_social_media_hours']*20, 
                        cmap='magma', alpha=0.6)
    
    plt.colorbar(points, label='Stress Level')
    plt.title("Screen Time Before Sleep vs Addiction Level", fontsize=16, fontweight='bold')
    plt.xlabel("Screen Time Before Sleep (Hours)", fontsize=12)
    plt.ylabel("Addiction Level", fontsize=12)
    plt.tight_layout()
    plt.savefig("Siloi_report/plots/plot_screentime_addiction_stress.png", dpi=300)
    #plt.show()

def plot_social_interaction_performance(df):
    """6. Social Interaction Level and Performance by Gender"""
    plt.figure(figsize=(12, 8))
    
    sns.boxenplot(data=df, x="social_interaction_level", y="academic_performance", 
                  hue="gender", palette="pastel", order=["low", "medium", "high"])
    
    plt.title("Social Interaction Level and Performance by Gender", fontsize=16, fontweight='bold')
    plt.xlabel("Social Interaction Level", fontsize=12)
    plt.ylabel("Academic Performance", fontsize=12)
    plt.tight_layout()
    plt.savefig("Siloi_report/plots/plot_social_interaction_performance.png", dpi=300)
    #plt.show()

if __name__ == "__main__":
    file_path = "Siloi_report/Teen_Mental_Health_Dataset.csv"
    df = load_data(file_path)

    # Create the plots folder
    os.makedirs("Siloi_report/plots", exist_ok=True)
    
    print("Generating professional plots...")
    
    # Plot graphs
    plot_age_stress_gender(df)
    plot_social_platform_stress(df)
    plot_sleep_performance_depression(df)
    plot_activity_anxiety_gender(df)
    plot_screentime_addiction_stress(df)
    plot_social_interaction_performance(df)
    
    print("All plots have been saved in the 'Siloi_report/plots/' folder.")

