# src/predict.py
import pandas as pd
import os
import joblib
from preprocessing import preprocess_data

def predict_new(csv_path, output_path="data/output/predictions.csv"):
    """
    Predict Laptop Prices from a new CSV file
    """
    # Load sample
    df = pd.read_csv(csv_path)
    
    # Preprocess
    df_processed = preprocess_data(df)
    
    # Load trained model
    if not os.path.exists("models/final_model.pkl"):
        raise FileNotFoundError("Trained model not found! Please run train.py first.")
    
    model = joblib.load("models/final_model.pkl")
    
    # Predict
    predictions = model.predict(df_processed)
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed['Predicted_Price'] = predictions
    df_processed.to_csv(output_path, index=False)
    
    print(f"✅ Predictions saved at {output_path}")
    return df_processed

if __name__ == "__main__":
    sample_csv = r"C:\dataset\Laptop-Price-Prediction\sample_input\sample_test.csv"
    predict_new(sample_csv)
