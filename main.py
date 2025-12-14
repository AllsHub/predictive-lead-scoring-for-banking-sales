from fastapi import FastAPI, HTTPException, UploadFile, File
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
            label = "HIGH_PRIORITY"
        elif prob >= THRESH_TIER_2:
            label = "MEDIUM_PRIORITY"
        else:
            label = "STANDARD_PRIORITY"

        # 4. Return Data Bersih
        return {
            "prediction": 1 if prob >= THRESH_TIER_2 else 0, # Opsional
            "score": prob,          # Untuk sorting
            "label_code": label,    # Untuk pemetaan
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check file type
    if not (file.filename.endswith('.csv') or file.filename.endswith(('.xlsx', '.xls'))):
        raise HTTPException(status_code=400, detail="File must be CSV or Excel (.csv, .xlsx, .xls)")
    
    try:
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
        
        # Rename columns to match model expectations
        rename_map = {
            'emp_var_rate': 'emp.var.rate',
            'cons_price_idx': 'cons.price.idx',
            'cons_conf_idx': 'cons.conf.idx',
            'nr_employed': 'nr.employed'
        }
        df = df.rename(columns=rename_map)
        
        # Process each row
        results = []
        for idx, row in df.iterrows():
            try:
                df_input = pd.DataFrame([row.to_dict()])
                prob = float(model.predict_proba(df_input)[0, 1])
                
                # Thresholds for prioritization
                THRESH_TIER_1 = 0.2841  # Top 10% (High Priority)
                THRESH_TIER_2 = 0.0685  # Top 30% (Medium Priority)
                
                if prob >= THRESH_TIER_1:
                    label = "HIGH_PRIORITY"
                elif prob >= THRESH_TIER_2:
                    label = "MEDIUM_PRIORITY"
                else:
                    label = "STANDARD_PRIORITY"
                
                results.append({
                    "row_index": idx,
                    "prediction": 1 if prob >= THRESH_TIER_2 else 0,
                    "score": prob,
                    "label_code": label
                })
            except Exception as e:
                results.append({
                    "row_index": idx,
                    "error": str(e)
                })
        
        return {"results": results, "total_processed": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))