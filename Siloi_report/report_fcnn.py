from calendar import day_abbr

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

def data_load_clean(data_file):
    
    # Load all data
    data = pd.read_csv(data_file) 

    # 1. Convert ALL text columns (categorical data) into numbers
    data['gender'] = data['gender'].map({'male': 0.0, 'female': 1.0})
    
    # New mappings for the other text-based features
    data['platform_usage'] = data['platform_usage'].map({'Instagram': 0.0, 'TikTok': 1.0, 'Both': 2.0})
    data['social_interaction_level'] = data['social_interaction_level'].map({'low': 0.0, 'medium': 1.0, 'high': 2.0})

    # 2. Convert the data into floats (using parentheses instead of brackets)
    data = data.astype(float)

    return data


class ReportDataset(Dataset):

    """
    data_file: csv file path
    features: list of features names for training the model
    labels: the name of the feature to use as label
    transform: list of tranformations to apply to the samples
    
    """

    def __init__(self, data_file, features, labels, transform=None):
        
        # Load and clean data from strings
        data = data_load_clean(data_file)

        # I divide samples from labels and transform everything to torch tensors
        self.samples = torch.tensor(data[features].values, dtype=torch.float32) # load features samples
        self.labels = torch.tensor(data[labels].values, dtype=torch.float32) # load the labels
        self.transform = transform # Needed transformations of both samples and labels

    def __len__(self):
        return len(self.labels) # Computes the number of samples
    
    def __getitem__(self, index):
        
        sample = self.samples[index]
        label = self.labels[index]
        
        # If defined, does the needed transformation to the sample
        if self.transform:
            sample = self.transform(sample)

        return sample, label # Return the sample and the label corresponding to the index

class TeenNet(nn.Module):
    
    def __init__(self):
        super(TeenNet, self).__init__()

    def forward(self, x):
        pass


# Data file path
datafile = r'Siloi_report/Teen_Mental_Health_Dataset.csv'

# The following lists are only one of many examples I can do
features = ["age","gender","daily_social_media_hours","platform_usage","sleep_hours",
            "screen_time_before_sleep","academic_performance",
            "physical_activity","social_interaction_level"]
labels = "depression_label"

# Define the dataset
teendataset = ReportDataset(datafile, features, labels, None)

# Defines the number of training and test samples 
total_samples = len(teendataset)
training_samples = int(0.8 * total_samples)
test_samples = total_samples - training_samples

# Sets the seed (if we want we can remove it)
generator = torch.Generator().manual_seed(100) 

# Splits the dataset into training and test sets
training_set, test_set = random_split(teendataset, [training_samples, test_samples], generator)

# Defines the dataloader for both training and test sets
train_dataloader = DataLoader(dataset=training_set, batch_size=40, shuffle=True)
test_dataloader = DataLoader(dataset=test_set, batch_size=40, shuffle=False)