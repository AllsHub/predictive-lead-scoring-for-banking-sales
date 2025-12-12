# train_model.py
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# 1. Import Transformers
from transformers import BankFeatureEngineer 

# 2. Load Data & Cleaning
print("Loading Data...")
df = pd.read_csv("bank-additional-full.csv", sep=';') 
df = (df.rename(columns=str.strip)
      .replace('unknown', np.nan)
      .assign(y=lambda x: x['y'].map({'yes': 1, 'no': 0}))
      .drop(columns=['duration'], errors='ignore')
      .drop_duplicates()
      .reset_index(drop=True))

X = df.drop(columns='y')
y = df['y']

# 3. Setup Pipeline
original_num_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                     'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
new_eng_cols = ['is_new_customer', 'high_contact_pressure', 'market_condition', 'life_stage']
num_cols = original_num_cols + new_eng_cols
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
            'contact', 'month', 'day_of_week', 'poutcome']

preprocessing_pipeline = Pipeline(steps=[
    ('feature_engineering', BankFeatureEngineer()),
    ('col_transformer', ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ],
        remainder='drop'
    ))
])

# 4. Model Training
# Memasukkan params hasil run Optuna dari notebook
best_params = {
    'n_estimators': 640, 'max_depth': 5, 'learning_rate': 0.030367678961434955,
    'subsample': 0.6990159226216422, 'colsample_bytree': 0.8209003953047949,
    'gamma': 3.4818717634031415, 'reg_alpha': 1.2769947614068164,
    'reg_lambda': 2.7026086870945445, 'scale_pos_weight': 1.0715062629096923,
    'n_jobs': -1, 'random_state': 42
}

xgb_best = XGBClassifier(**best_params)
calibrated_model = CalibratedClassifierCV(estimator=xgb_best, method='sigmoid', cv=5)

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessing_pipeline),
    ('model', calibrated_model)
])

print("Training Model...")
final_pipeline.fit(X, y)

# 5. Save ke .pkl
joblib.dump(final_pipeline, 'model_deposito_siap_pakai.pkl')
print("File 'model_deposito_siap_pakai.pkl' berhasil dibuat.")