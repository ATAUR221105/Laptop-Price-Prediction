# src/train.py
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import preprocess_data

def train_model(csv_path):
    # ----------------------------
    # Load and preprocess data
    # ----------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = preprocess_data(df)
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ----------------------------
    # Train XGBoost Regressor
    # ----------------------------
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # ----------------------------
    # Evaluation Metrics
    # ----------------------------
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    
    print("----- XGBoost Regression Results -----")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2 Score (Test): {r2:.4f}")
    print(f"5-Fold CV Mean R2: {cv_r2:.4f}")
    
    # ----------------------------
    # Save metrics
    # ----------------------------
    results_folder = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_folder, exist_ok=True)
    metrics_file = os.path.join(results_folder, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("----- XGBoost Regression Results -----\n")
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"R2 Score (Test): {r2:.4f}\n")
        f.write(f"5-Fold CV Mean R2: {cv_r2:.4f}\n")
    print(f"✅ Metrics saved at {metrics_file}")
    
    # ----------------------------
    # Save Model
    # ----------------------------
    models_folder = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_folder, exist_ok=True)
    model_file = os.path.join(models_folder, 'final_model.pkl')
    joblib.dump(model, model_file)
    print(f"✅ Model saved at {model_file}")
    
    # ----------------------------
    # Feature Importance Plot
    # ----------------------------
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "feature_importance.png"))
    plt.close()
    
    # ----------------------------
    # Residual Plot
    # ----------------------------
    residuals = y_test - y_pred
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Price")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "confusion_matrix.png"))
    plt.close()
    
    # ----------------------------
    # Predicted vs Actual Plot
    # ----------------------------
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual Price")
    plt.tight_layout()
    plt.show()
    
    # ----------------------------
    # Save Test Predictions CSV
    # ----------------------------
    output_folder = os.path.join(os.getcwd(), 'data', 'output')
    os.makedirs(output_folder, exist_ok=True)
    output_csv = os.path.join(output_folder, 'test_predictions.csv')
    
    X_test_copy = X_test.copy()
    X_test_copy['Actual_Price'] = y_test
    X_test_copy['Predicted_Price'] = y_pred
    X_test_copy.to_csv(output_csv, index=False)
    print(f"✅ Test predictions saved at {output_csv}")
    
    return model

if __name__ == "__main__":
    csv_path = r"C:\dataset\Laptop-Price-Prediction\data\raw\Laptop_price.csv"
    train_model(csv_path)
