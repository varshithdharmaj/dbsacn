import streamlit as st
import numpy as np
import pickle
import os

# Load the trained model using absolute path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parkinson_model.pkl')

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Optional: Load the scaler if you used StandardScaler or MinMaxScaler during training
try:
    SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    with open(SCALER_PATH, 'rb') as file:
        scaler = pickle.load(file)
    scaler_loaded = True
except FileNotFoundError:
    scaler_loaded = False

# Streamlit UI
st.set_page_config(page_title="Parkinson's Disease Detection", page_icon="🧠")
st.title("🧠 Parkinson’s Disease Detection App")
st.markdown("Upload voice features to detect Parkinson’s Disease.")

# Input fields
fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, step=0.1)
fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, step=0.1)
flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, step=0.1)
jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.001)
rap = st.number_input("MDVP:RAP", min_value=0.0, step=0.001)
ppe = st.number_input("PPE", min_value=0.0, step=0.001)

# Prepare input
user_input = np.array([[fo, fhi, flo, jitter_percent, rap, ppe]])

# Predict
if st.button("Predict"):
    try:
        # Apply scaling if scaler is available
        if scaler_loaded:
            user_input_scaled = scaler.transform(user_input)
            prediction = model.predict(user_input_scaled)
        else:
            prediction = model.predict(user_input)

        if prediction[0] == 1:
            st.error("⚠️ The patient is likely to have Parkinson’s Disease.")
        else:
            st.success("✅ The patient is unlikely to have Parkinson’s Disease.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
