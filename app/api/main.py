from fastapi import FastAPI, UploadFile, File
import pandas as pd
import shutil
import os
from supervised.automl import AutoML

app = FastAPI(title="AutoML-Studio API", description="Production API for automated model predictions")

# Path where your trained models are stored
MODEL_PATH = "../../my_automl" 

@app.get("/")
def read_root():
    return {"message": "AutoML-Studio API is online. Ready for predictions."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Load the incoming data
    df = pd.read_csv(file.file)
    
    # 2. Load the best model from the Studio
    automl = AutoML(results_path=MODEL_PATH)
    
    # 3. Generate predictions
    predictions = automl.predict_all(df)
    
    # 4. Return results as JSON
    return predictions.to_dict(orient="records")