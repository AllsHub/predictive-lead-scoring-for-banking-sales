# FILE: transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df_eng = X.copy()
        
        # 1. is_new_customer
        df_eng['is_new_customer'] = (df_eng['pdays'] == 999).astype(int)
        
        # 2. high_contact_pressure
        df_eng['high_contact_pressure'] = (df_eng['campaign'] > 4).astype(int)
        
        # 3. market_condition
        df_eng['market_condition'] = df_eng['euribor3m'] * df_eng['cons.conf.idx']
        
        # 4. life_stage
        conditions = [
            (df_eng['age'] < 30),
            (df_eng['age'] >= 30) & (df_eng['age'] <= 60),
            (df_eng['age'] > 60)
        ]
        df_eng['life_stage'] = np.select(conditions, [0, 1, 2], default=1)
        return df_eng