import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Parkinson's Detection", layout="centered")

st.title("🧠 Parkinson's Disease Detection App")

model_6_path = "model_6.pkl"
scaler_6_path = "scaler_6.pkl"
model_22_path = "model_22.pkl"
scaler_22_path = "scaler_22.pkl"

# Load models & scalers if available
model_6 = pickle.load(open(model_6_path, "rb")) if os.path.exists(model_6_path) else None
scaler_6 = pickle.load(open(scaler_6_path, "rb")) if os.path.exists(scaler_6_path) else None
model_22 = pickle.load(open(model_22_path, "rb")) if os.path.exists(model_22_path) else None
scaler_22 = pickle.load(open(scaler_22_path, "rb")) if os.path.exists(scaler_22_path) else None

model_choice = st.radio("Choose model:", ["6 Features", "22 Features"])

if model_choice == "6 Features":
    st.subheader("Enter 6 Features:")
    fo = st.number_input("Average vocal fundamental frequency (fo)", value=135.0)
    fhi = st.number_input("Maximum vocal fundamental frequency (fhi)", value=159.0)
    flo = st.number_input("Minimum vocal fundamental frequency (flo)", value=116.0)
    jitter_percent = st.number_input("Jitter (%)", value=0.002)
    rap = st.number_input("Relative amplitude perturbation (RAP)", value=0.0015)
    ppe = st.number_input("Pitch period entropy (PPE)", value=0.25)

    features = [fo, fhi, flo, jitter_percent, rap, ppe]
    if st.button("Predict"):
        if model_6 is None or scaler_6 is None:
            st.error("❌ 6-feature model or scaler file not found. Please ensure 'model_6.pkl' and 'scaler_6.pkl' are in the app folder.")
        else:
            X = scaler_6.transform([features])
            pred = model_6.predict(X)[0]
            st.success(f"🔍 Prediction: {'Parkinson\'s Likely' if pred == 1 else 'Healthy (No Parkinson\'s)'}")

else:
    st.subheader("Enter 22 Features:")
    features_22 = []
    default_values = [
        119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370, 0.00554, 0.01109, 0.04374,
        0.33371, -4.81303, 0.28144, 2.30144, 0.28465, 0.18669, 2.30144, 0.03130, 21.033,
        0.414783, 0.593763, 0.631299, 0.225
    ]

    labels_22 = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    for label, default in zip(labels_22, default_values):
        val = st.number_input(label, value=float(default))
        features_22.append(val)

    if st.button("Predict"):
        if model_22 is None or scaler_22 is None:
            st.error("❌ 22-feature model or scaler file not found. Please ensure 'model_22.pkl' and 'scaler_22.pkl' are in the app folder.")
        else:
            X = scaler_22.transform([features_22])
            pred = model_22.predict(X)[0]
            st.success(f"🔍 Prediction: {'Parkinson\'s Likely' if pred == 1 else 'Healthy (No Parkinson\'s)'}")
