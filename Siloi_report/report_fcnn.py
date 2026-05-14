import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


########################################################################
# CUSTOM FUNCTIONS NEEDED FOR IN MAIN CODE
########################################################################

# Load the dataset and clean it from unwanted data types
def data_load_clean(data_file):
    
    # Load all data
    data = pd.read_csv(data_file) 

    # 1. Convert ALL text columns (categorical data) into numbers
    data['gender'] = data['gender'].map({'male': 0.0, 'female': 1.0})
    data['platform_usage'] = data['platform_usage'].map({'Instagram': 0.0, 'TikTok': 1.0, 'Both': 2.0})
    data['social_interaction_level'] = data['social_interaction_level'].map({'low': 0.0, 'medium': 1.0, 'high': 2.0})

    # 2. Convert the data into floats (using parentheses instead of brackets)
    data = data.astype(float)

    return data

def train_test_split(dataset: Dataset, train_ratio: float = 0.8, seed: int | None = 100):
    
    # Defines the number of training and test samples 
    total_samples = len(dataset)
    training_samples = int(train_ratio * total_samples)
    test_samples = total_samples - training_samples

    if seed is not None:
        generator = torch.Generator().manual_seed(seed) 

        return random_split(dataset, [training_samples, test_samples], generator=generator)
    
    return random_split(dataset, [training_samples, test_samples])


# Train the model
def train_model(teenmodel, num_epochs, train_dataloader, criterion, optimizer):
    
    # Set the model in training 
    teenmodel.train()

    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(train_dataloader):

            # forward pass
            outputs = teenmodel(samples)
            loss = criterion(outputs, labels)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
def test_model(teenmodel, test_dataloader):

    teenmodel.eval()

    with torch.no_grad():
        n_correct = 0 
        n_samples = 0
        for samples, labels in test_dataloader:
            outputs = teenmodel(samples) 
            predictions = torch.argmax(outputs, dim=1)
            n_correct += (predictions == labels).sum().item()
            n_samples += labels.shape[0]

    accuracy = 100.0 * (n_correct / n_samples)
    print(f'The current model accuracy is: {accuracy} %')

     
########################################################################
# DATASET AND NEURAL NETWORK CLASSES
########################################################################

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
        self.labels = torch.tensor(data[labels].values, dtype=torch.long) # load the labels
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

class EX1Net(nn.Module):
    
    def __init__(self, input_size, num_classes, p_dropout):
        super(EX1Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)    # First layer - Input
        self.fc2 = nn.Linear(20, 20)            # Second layer - hidden
        self.fc3 = nn.Linear(20, num_classes)   # Third layer - output
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        
        # First layer pass
        out = self.fc1(x) 
        out = F.relu(out)
        out = self.dropout(out)

        # Second layer pass
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Third layer pass
        out = self.fc3(out)

        return out

########################################################################
# MAIN CODE
########################################################################

# Data file path
datafile = r'Siloi_report/Teen_Mental_Health_Dataset.csv'

all_features = np.array(["age","gender","daily_social_media_hours","platform_usage",
                     "sleep_hours","screen_time_before_sleep","academic_performance",
                     "physical_activity","social_interaction_level","stress_level",
                     "anxiety_level","addiction_level","depression_label"])

# The following lists are only one of many examples I can do
sample_features = all_features[:9]
labels = all_features[12] 

num_features, num_classes = len(sample_features), 10

# Define the dataset
teendataset = ReportDataset(datafile, sample_features, labels, None)

# Split training and test datasets
training_set, test_set = train_test_split(teendataset)

# Defines the dataloader for both training and test sets
train_dataloader = DataLoader(dataset=training_set, batch_size=40, shuffle=True)
test_dataloader = DataLoader(dataset=test_set, batch_size=40, shuffle=False)

########################################################################

# Model definition and setting hyperparameters

p_drop, learning_rate, num_epochs = 0.2, 0.01, 100

teenmodel = EX1Net(num_features, num_classes, p_drop)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(teenmodel.parameters(), lr=learning_rate)

########################################################################

# Training phase

train_model(teenmodel, num_epochs, train_dataloader, criterion, optimizer)

# Test phase

test_model(teenmodel, test_dataloader)
