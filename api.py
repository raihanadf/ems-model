from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from treatment import Treatment
import pandas as pd
import pickle
import subprocess

app = FastAPI()


def predict(treatment: Treatment):
    # load the model from pickle files
    with open("pickles/ems_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("pickles/label_encoding.pkl", "rb") as f:
        le = pickle.load(f)

    with open("pickles/scaler_encoding.pkl", "rb") as f:
        scaler = pickle.load(f)

    # encode the species species
    species = le.transform([treatment.species])

    # normalize the input data
    transformed_data = scaler.transform(
        pd.DataFrame(
            {
                "soakDuration": [treatment.soakDuration],
                "lowestTemp": [treatment.lowestTemp],
                "highestTemp": [treatment.highestTemp],
            }
        )
    )

    data = pd.DataFrame(
        {
            "species": [species],
            "emsConcentration": [treatment.emsConcentration],
            "soakDuration": [transformed_data[0][0]],
            "lowestTemp": [transformed_data[0][1]],
            "highestTemp": [transformed_data[0][2]],
        }
    )

    prediction_prob = model.predict_proba(data)[0]
    prediction = model.predict(data)[0]

    confidence_score = prediction_prob[prediction] * 100

    result = {
        "result": int(prediction),
        "confidence_score": float(confidence_score),
        "success_rate": float(prediction_prob[1] * 100),
    }

    return JSONResponse(content=jsonable_encoder(result))


@app.post("/process")
def process(treatment: Treatment):
    return predict(treatment)


@app.get("/species")
def get_species():
    df = pd.read_csv("csv/ems_data.csv")
    unique_values = df.iloc[:, 0].drop_duplicates().tolist()

    return JSONResponse(content=unique_values)


@app.get("/retrain-model")
def retrain_model():
    try:
        # run the model.py script and capture output
        result = subprocess.run(
            ["python", "model.py"],
            check=True,
            capture_output=True,  # captures stdout and stderr
            text=True,  # decodes the output into strings
        )
        # extract and format the output
        output = result.stdout.strip()
        return {"message": "Retraining succeeded!", "details": output}
    except subprocess.CalledProcessError as e:
        # handle the case where the script fails
        return {"message": "Retraining failed!", "error": str(e)}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # check if the uploaded file is a csv
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

        # save the uploaded file to csv/ems_data.csv (overwrite)
        file_path = "csv/ems_data.csv"
        with open(file_path, "wb") as f:
            content = await file.read()  # read the uploaded file content
            f.write(content)  # write the content to the destination file

        # verify if the file is valid (optional: try loading it as a dataframe)
        try:
            pd.read_csv(
                file_path
            )  # this will raise an error if the file is not a valid csv
        except Exception:
            os.remove(file_path)  # remove invalid csv
            raise HTTPException(
                status_code=400, detail="Uploaded file is not a valid CSV."
            )

        return {"message": "File uploaded and overwritten successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-csv")
def get_csv():
    file_path = "csv/ems_data.csv"

    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Convert the DataFrame to a list of dictionaries (JSON-like format)
        data = df.to_dict(orient="records")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")
