# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Preprocess the Laptop Price dataset
    - Encode 'Brand' categorical column
    """
    df = df.copy()
    
    # Encode Brand
    le = LabelEncoder()
    df['Brand'] = le.fit_transform(df['Brand'])
    
    return df
