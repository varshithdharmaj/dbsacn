import streamlit as st
import numpy as np
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
st.markdown("Enter the values for each feature below:")

# Collect all inputs
input_features = []
for name in feature_names:
    value = st.number_input(name, step=0.001, format="%.6f")
    input_features.append(value)

input_data = np.array([input_features])

if st.button("Predict"):
    if scaler_loaded:
        input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ The person is likely to have Parkinson's disease.\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ The person is unlikely to have Parkinson's disease.\nProbability: {probability:.2%}")
