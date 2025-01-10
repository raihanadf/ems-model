import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load the data from .csv file (must be a csv)
    """
    try:
        df = pd.read_csv(file_path)
        print("CSV Loaded, Dataset shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Error: CSV File {file_path} not found!")
        return None

def preprocess_data(df):
    """
    Preprocess data: 
    - LabelEencoder to species (categorical)
    - MinMaxScaling for the rest (numerical)
    """
    # Create label encoder for species
    le = LabelEncoder()
    df['species_encoded'] = le.fit_transform(df['species'])
    
    # Create MinMax scaler for numerical features
    scaler = MinMaxScaler()
    numerical_columns = ['emsConcentration', 'soakDuration', 'lowestTemp', 'highestTemp']
    
    # Scale numerical features
    scaled_features = scaler.fit_transform(df[numerical_columns])
    
    # Create DataFrame with scaled features
    scaled_df = pd.DataFrame(scaled_features, columns=numerical_columns)
    
    # Add species_encoded to the scaled features
    scaled_df['species_encoded'] = df['species_encoded']
    
    # Prepare features and target
    X = scaled_df[['species_encoded', 'emsConcentration', 'soakDuration', 
                   'lowestTemp', 'highestTemp']]
    y = df['result']
    
    return X, y

def train_model(X, y):
    """
    Split data and train Random Forest model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        # max_depth=10,
        # min_samples_split=5,
        # min_samples_leaf=2,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    return rf_model, X_train, X_test, y_train, y_test

def main():
    print("Hello from ems-model!")

    # load from csv file
    df = load_data("ems_data.csv") 

    # preprocess
    X, y = preprocess_data(df)

    # train
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    y_pred = model.predict(X_test)
    y_test = np.array(y_test)

    # print classification report
    # print(y_pred)
    # print(y_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
