from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from treatment import Treatment
import pandas as pd
import pickle

app = FastAPI()

def predict(treatment: Treatment):
    # load the model from pickles file
    with open('pickles/ems_model.pkl', 'rb') as f:
      model = pickle.load(f)

    with open('pickles/label_encoding.pkl', 'rb') as f:
      le = pickle.load(f)

    with open('pickles/scaler_encoding.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # encode the species species
    species = le.transform([treatment.species])

    # normalize the input data
    transformed_data = scaler.transform(pd.DataFrame({
        "soakDuration": [treatment.soakDuration],
        "lowestTemp": [treatment.lowestTemp],
        "highestTemp": [treatment.highestTemp],
    }))

    data = pd.DataFrame({
        "species" : [species],
        "emsConcentration" : [treatment.emsConcentration],
        "soakDuration": [transformed_data[0][0]],
        "lowestTemp": [transformed_data[0][1]],
        "highestTemp": [transformed_data[0][2]],
    })

    prediction_prob = model.predict_proba(data)[0] 
    prediction = model.predict(data)[0]

    confidence_score = prediction_prob[prediction] * 100
    result = {
        "result": int(prediction), 
        "confidence_score": float(confidence_score)
    }

    return JSONResponse(content=jsonable_encoder(result))

@app.post("/process")
def process(treatment: Treatment):
    return predict(treatment)
