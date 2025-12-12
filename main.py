from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from transformers import BankFeatureEngineer
import os

app = FastAPI(title="Bank Telemarketing Prediction AI")

# Load Model
model = None
model_filename = "model_deposito_siap_pakai.pkl"

if os.path.exists(model_filename):
    try:
        model = joblib.load(model_filename)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

# Schema Data
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

@app.get("/")
def home():
    return {"status": "AI Service Ready", "version": "1.0"}

@app.post("/predict")
def predict_deposit(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Prepare Data
        input_data = data.dict()
        rename_map = {
            'emp_var_rate': 'emp.var.rate',
            'cons_price_idx': 'cons.price.idx',
            'cons_conf_idx': 'cons.conf.idx',
            'nr_employed': 'nr.employed'
        }
        df_input = pd.DataFrame([input_data]).rename(columns=rename_map)
        
        # 2. Dapatkan Probabilitas Murni (0.0 - 1.0)
        prob = float(model.predict_proba(df_input)[0, 1])
        
        # 3. Logika Tiering
        # Hasil perhitungan di notebook
        THRESH_TIER_1 = 0.2841  # Batas untuk Top 10% (High Priority)
        THRESH_TIER_2 = 0.0685  # Batas untuk Top 30% (Medium Priority)

        # Penentuan Label Abstrak
        if prob >= THRESH_TIER_1:
            tier = "TIER_1"
            label = "HIGH_PRIORITY"
        elif prob >= THRESH_TIER_2:
            tier = "TIER_2"
            label = "MEDIUM_PRIORITY"
        else:
            tier = "TIER_3"
            label = "STANDARD_PRIORITY"

        # 4. Return Data Bersih
        return {
            "prediction": 1 if prob >= THRESH_TIER_2 else 0, # Opsional
            "score": prob,          # Untuk sorting
            "tier": tier,           # Untuk logic system
            "label_code": label,    # Untuk pemetaan frontend
            "description": f"Probability: {prob:.2%}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))