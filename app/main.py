from fastapi import FastAPI
from app.model import Mushroom
import joblib
import pandas as pd
import numpy as np
from nn_model import build_model 
from wrapper import DataProcessorTransformer 

app = FastAPI()

# carica modello (con pipeline inclusa)
lr_model= joblib.load("models/lr_model_pipe.joblib")
xgb_model = joblib.load("models/xgb_model_pipe.joblib")
nn_model = joblib.load("models/nn_model_pipe.joblib")
columns = joblib.load("models/columns.joblib")
col_bin = joblib.load("models/col_bin.joblib")
col_hot = joblib.load("models/col_hot.joblib")

# -------------------------
# TEST ENDPOINT
# -------------------------

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}


# -------------------------
# PREDICTION ENDPOINT
# -------------------------
@app.post("/predict/lr")
def predict(data: Mushroom):
    df = pd.DataFrame([data.model_dump()]) 
    df = df[columns]

    pred = lr_model.predict(df)[0]
    
    if isinstance(pred, np.generic):
        pred = pred.item()

    return {"Category": pred}



@app.post("/predict/xgb")
def predict(data: Mushroom):
    df = pd.DataFrame([data.model_dump()]) 
    df = df[columns]

    pred = xgb_model.predict(df)[0]
    
    if isinstance(pred, np.generic):
        pred = pred.item()

    return {"Category": pred}



@app.post("/predict/nn")
def predict(data: Mushroom):
    df = pd.DataFrame([data.model_dump()]) 
    df = df[columns]

    pred = nn_model.predict_proba(df)[0]
    
    if isinstance(pred, np.generic):
        pred = pred.item()

    return {"Category": pred}

 
    
# TO RUN ON TERMINAL DO : uvicorn app.main:app --reload