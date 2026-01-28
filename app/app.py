# app/app.py
import streamlit as st
import pandas as pd
import joblib
import os
import sys

# preprocessing import
sys.path.append(os.path.abspath("src"))
from preprocessing import preprocess_data


# ----------------------------
# Load trained model
# ----------------------------
MODEL_PATH = "models/final_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Laptop Price Prediction", layout="centered")

st.title("💻 Laptop Price Prediction")
st.write("Enter laptop specifications to predict price")


# ----------------------------
# User Inputs (MATCH TRAINING FEATURES)
# ----------------------------
brand = st.selectbox(
    "Brand",
    ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "Other"]
)

processor_speed = st.number_input(
    "Processor Speed (GHz)",
    min_value=1.0,
    max_value=6.0,
    step=0.1
)

ram_size = st.number_input(
    "RAM Size (GB)",
    min_value=2,
    max_value=64,
    step=2
)

storage_capacity = st.number_input(
    "Storage Capacity (GB)",
    min_value=128,
    max_value=4096,
    step=128
)

screen_size = st.number_input(
    "Screen Size (inch)",
    min_value=10.0,
    max_value=18.0,
    step=0.1
)

weight = st.number_input(
    "Weight (kg)",
    min_value=0.8,
    max_value=5.0,
    step=0.1
)


# ----------------------------
# Predict Button
# ----------------------------
if st.button("🔮 Predict Price"):
    try:
        # EXACT feature names as training
        input_df = pd.DataFrame([{
            "Brand": brand,
            "Processor_Speed": processor_speed,
            "RAM_Size": ram_size,
            "Storage_Capacity": storage_capacity,
            "Screen_Size": screen_size,
            "Weight": weight
        }])

        # Preprocess
        input_processed = preprocess_data(input_df)

        # Prediction
        prediction = model.predict(input_processed)[0]

        st.success(f"💰 Predicted Laptop Price: **{prediction:,.0f}**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
