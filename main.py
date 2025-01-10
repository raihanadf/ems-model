import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
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

def main():
    print("Hello from ems-model!")
    load_data("ems_data.csv") # load from csv file

if __name__ == "__main__":
    main()
