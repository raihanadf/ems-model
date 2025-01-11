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
        X, y, test_size=0.2, random_state=64
    )
    
    # Initialize and train model
    # use hitchhiker's guide to the galaxy which is 42
    # if u just wanted to, say a random number
    #
    # Deep Thought had been built by its creators to give the answer to the "Ultimate Question of Life, the Universe, and Everything", which, after eons of calculations, was given simply as "42". 
    #
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=17,
        # min_samples_split=5,
        # min_samples_leaf=2,
        criterion='gini',
        random_state=64
    )
    
    rf_model.fit(X_train, y_train)
    
    return rf_model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    y_test = np.array(y_test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # plt.show()

    print(f"Model Accuracy Score: {round(model.score(X_test, y_test) * 100,1)}%")

    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']).rename_axis(index={'Actual': 'Actual'}, columns={'Predicted': 'Predicted'})
    .set_axis(['Fail rate', 'Success rate'], axis=0)
    .set_axis(['Fail rate', 'Success rate'], axis=1))

def main():
    print("Hello from ems-model!")

    # load from csv file
    df = load_data("ems_data.csv") 

    # preprocess
    X, y = preprocess_data(df)

    # train
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # evaluate
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
