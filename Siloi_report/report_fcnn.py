from email import header

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split

class ReportDataset(Dataset):

    """
    data_file: csv file path
    features: list of features names for training the model
    labels: the name of the feature to use as label
    transform: list of tranformations to apply to the samples
    
    """

    def __init__(self, data_file, features, labels, transform=None):
        
        data = pd.read_csv(data_file) # Load all data

        # I divide samples from labels and transform everything to torch tensors

        self.samples = torch.tensor(data[features].to_numpy(), dtype=torch.float32) # load features samples
        self.labels = torch.tensor(data[labels].to_numpy(), dtype=torch.long) # load the labels
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


datafile = r'Siloi_report/Teen_Mental_Health_Dataset.csv'

# The following lists are only one of many examples I can do
features = ["age","gender","daily_social_media_hours","platform_usage","sleep_hours",
            "screen_time_before_sleep","academic_performance",
            "physical_activity","social_interaction_level"]
labels = "depression_label"

# Define the dataset
teendataset = ReportDataset(datafile, features, labels, None)
