import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

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
    # check if pickles directory exist
    if not os.path.exists("pickles"):
        os.makedirs("pickles")

    # create label encoder for species and save it
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])

    with open('pickles/label_encoding.pkl', 'wb') as f:
      pickle.dump(le, f)

    # create minmax scaler for numerical features
    scaler = MinMaxScaler()
    numerical_columns = ['soakDuration', 'lowestTemp', 'highestTemp']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    scaled_features = df[numerical_columns]

    with open('pickles/scaler_encoding.pkl', 'wb') as f:
      pickle.dump(scaler, f)

    # prepare features and target
    X = df[['species', 'emsConcentration', 'soakDuration', 
                   'lowestTemp', 'highestTemp']]
    y = df['result']

    return X, y

def train_model(X, y):
    """
    Split data and train Random Forest model
    """
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=64
    )

    # initialize and train model
    # use hitchhiker's guide to the galaxy which is 42
    # if u just wanted to, say a random number
    #
    # "Deep Thought had been built by its creators to give the answer to the "Ultimate Question of Life, the Universe, and Everything", which, after eons of calculations, was given simply as "42"." - Wikipedia about hitchiker's guide to the galaxy
    #
    rf_model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        random_state=64
    )

    rf_model.fit(X_train, y_train)

    with open('pickles/ems_model.pkl', 'wb') as f:
      pickle.dump(rf_model, f)

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

    print(f"Model Accuracy Score: {round(model.score(X_test, y_test) * 100,1)}% \n")

    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    .rename_axis(index={'Actual': 'Actual'}, columns={'Predicted': 'Predicted'})
    .set_axis(['Fail rate', 'Success rate'], axis=0)
    .set_axis(['Fail rate', 'Success rate'], axis=1))

def feature_importance(model):
    """
    Plot feature importance
    Check the importance of each feature
    """
    feature_names = ['Species', 'EMS Concentration', 'Soak Duration', 'Lowest Temperature', 'Highest Temperature']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')

def test_load():
    # load the model from pickles file
    with open('pickles/ems_model.pkl', 'rb') as f:
      model = pickle.load(f)

    with open('pickles/label_encoding.pkl', 'rb') as f:
      le = pickle.load(f)

    with open('pickles/scaler_encoding.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # test example input
    species = "Arabidopsis"
    emsConcentration = 1.3
    soakDuration = 600
    lowestTemp = 24
    highestTemp = 30

    # encode the species species
    species = le.transform([species])

    # normalize the input data
    input_data = scaler.transform(pd.DataFrame({
        # "emsConcentration": [emsConcentration],
        "soakDuration": [soakDuration],
        "lowestTemp": [lowestTemp],
        "highestTemp": [highestTemp],
    }))

    data = pd.DataFrame({
        "species" : [species],
        "emsConcentration" : [emsConcentration],
        "soakDuration": [input_data[0][0]],
        "lowestTemp": [input_data[0][1]],
        "highestTemp": [input_data[0][2]],
    })

    # make a prediction
    prediction_prob = model.predict_proba(data)
    print(prediction_prob[0][0] * 100)

def main():
    print("Hello from ems-model!")

    # load from csv file
    df = load_data("csv/ems_data.csv") 

    # preprocess
    X, y = preprocess_data(df)

    # train
    model, _, X_test, _, y_test = train_model(X, y)

    # evaluate
    evaluate_model(model, X_test, y_test)

    # feature importance
    feature_importance(model)

    # show evaluation on confusion matrix and feature importance
    # plt.show()

    test_load()

if __name__ == "__main__":
    main()
