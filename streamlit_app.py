import streamlit as st
import numpy as np
import pickle

# Load models and scalers
try:
    model_6 = pickle.load(open("model_6.pkl", "rb"))
    scaler_6 = pickle.load(open("scaler_6.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ 6-feature model or scaler file not found. Please ensure 'model_6.pkl' and 'scaler_6.pkl' are in the app folder.")
    st.stop()

try:
    model_22 = pickle.load(open("model_22.pkl", "rb"))
    scaler_22 = pickle.load(open("scaler_22.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ 22-feature model or scaler file not found. Please ensure 'model_22.pkl' and 'scaler_22.pkl' are in the app folder.")
    st.stop()

# Feature names
features_6 = ['fo', 'fhi', 'flo', 'jitter_percent', 'rap', 'ppe']
features_22 = [
    'fo', 'fhi', 'flo', 'jitter_percent', 'jitter_abs', 'rap', 'ppq', 'ddp',
    'shimmer', 'shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 'apq', 'dda',
    'nhr', 'hnr', 'rpde', 'dfa', 'spread1', 'spread2', 'ppe', 'mdvp_fv'
]

# App title
st.title("🧠 Parkinson's Disease Prediction App")

# Model selection
model_choice = st.radio("Select Prediction Model", ["6 Features", "22 Features"])

# 6-Feature Mode
if model_choice == "6 Features":
    st.subheader("🔢 Input 6 Voice Features")
    input_values = []
    for feature in features_6:
        value = st.number_input(f"{feature}", format="%.6f")
        input_values.append(value)

    if st.button("🔍 Predict (6 Features)"):
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler_6.transform(input_array)
        prediction = model_6.predict(input_scaled)[0]

        if prediction == 1:
            st.error("🔍 Prediction: Parkinson's Likely")
        else:
            st.success("✅ Prediction: Parkinson's Not Likely")

# 22-Feature Mode
elif model_choice == "22 Features":
    st.subheader("🔢 Input 22 Voice Features")
    input_values = []
    for feature in features_22:
        value = st.number_input(f"{feature}", format="%.6f")
        input_values.append(value)

    if st.button("🔍 Predict (22 Features)"):
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler_22.transform(input_array)
        prediction = model_22.predict(input_scaled)[0]

        if prediction == 1:
            st.error("🔍 Prediction: Parkinson's Likely")
        else:
            st.success("✅ Prediction: Parkinson's Not Likely")
