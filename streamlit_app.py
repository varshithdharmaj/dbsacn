import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parkinson_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Load scaler if used
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as s:
        scaler = pickle.load(s)
    scaler_loaded = True
else:
    scaler_loaded = False

# Feature list (used while training)
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
    'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# Streamlit UI
st.set_page_config(page_title="Parkinson's Prediction", layout="centered")
st.title("🧠 Parkinson’s Disease Prediction App")
st.markdown("Choose how you want to input the data:")

# Select input mode
input_mode = st.radio("Select Input Method", ["Manual Entry", "CSV Upload", "Paste Values"])

input_data = None

if input_mode == "Manual Entry":
    input_features = []
    for name in feature_names:
        value = st.number_input(name, step=0.001, format="%.6f")
        input_features.append(value)
    input_data = np.array([input_features])

elif input_mode == "CSV Upload":
    uploaded_file = st.file_uploader("Upload a CSV file with 1 row and 22 columns", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] != 22:
                st.error("CSV must contain exactly 22 columns.")
            else:
                input_data = df.values
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

elif input_mode == "Paste Values":
    text_input = st.text_area("Paste 22 comma-separated values", height=100)
    if text_input:
        try:
            values = [float(x.strip()) for x in text_input.split(",")]
            if len(values) != 22:
                st.error("You must enter exactly 22 values.")
            else:
                input_data = np.array([values])
        except ValueError:
            st.error("Please ensure all values are numeric and comma-separated.")

# Predict if data is ready
if input_data is not None and st.button("Predict"):
    if scaler_loaded:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Likely Parkinson's Disease Detected. Probability: {probability:.2%}")
    else:
        st.success(f"✅ Unlikely to have Parkinson's Disease. Probability: {probability:.2%}")
