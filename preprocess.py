import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """
    Load the dataset from the given file path
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the data for training
    """
    # Separate features and labels
    X = data.iloc[:, 1:].values  # All columns except the first one (labels)
    y = data.iloc[:, 0].values   # First column (labels)
    
    # Normalize the features
    X = X / 255.0
    
    # Reshape the data for CNN input
    X = X.reshape(-1, 28, 28, 1)
    
    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y, label_encoder

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test 