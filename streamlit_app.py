import streamlit as st
import numpy as np
import pickle
import os

# === Load the Model ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parkinson_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# === Load the Scaler (if exists) ===
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, 'rb') as s:
            scaler = pickle.load(s)
    except Exception as e:
        st.warning(f"⚠️ Scaler exists but couldn't be loaded: {e}")
else:
    st.info("ℹ️ No scaler found. Input will be used without scaling.")

# === Feature List ===
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
    'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# === Streamlit UI ===
st.set_page_config(page_title="Parkinson's Prediction", layout="centered")
st.title("🧠 Parkinson’s Disease Prediction App")
st.markdown("Please enter the following voice measurements:")

# === Collect Input Features ===
input_features = []
for name in feature_names:
    value = st.number_input(name, step=0.001, format="%.6f")
    input_features.append(value)

input_data = np.array([input_features])  # shape: (1, 22)

# === Predict Button ===
if st.button("🔍 Predict"):
    try:
        # Apply scaler if available
        if scaler:
            input_data = scaler.transform(input_data)
        
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

        if prediction[0] == 1:
            st.error(f"⚠️ Likely Parkinson's Disease Detected.\nProbability: {probability:.2%}" if probability else "⚠️ Likely Parkinson's Disease Detected.")
        else:
            st.success(f"✅ Unlikely to have Parkinson's Disease.\nProbability: {probability:.2%}" if probability else "✅ Unlikely to have Parkinson's Disease.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
