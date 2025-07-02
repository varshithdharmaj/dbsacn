
import streamlit as st
import numpy as np
import pickle

# Load the trained model
import os
import pickle

# Load the model safely using absolute path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parkinson_model.pkl')

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)


st.set_page_config(page_title="Parkinson's Disease Detection", page_icon="🧠")
st.title("🧠 Parkinson’s Disease Detection App")

st.markdown("Upload voice features to detect Parkinson’s Disease.")

# Input fields (add more if needed)
fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, step=0.1)
fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, step=0.1)
flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, step=0.1)
jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.001)
rap = st.number_input("MDVP:RAP", min_value=0.0, step=0.001)
ppe = st.number_input("PPE", min_value=0.0, step=0.001)

# Prepare input for prediction
input_data = np.array([[fo, fhi, flo, jitter_percent, rap, ppe]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Parkinson’s Disease.")
    else:
        st.success("✅ The patient is unlikely to have Parkinson’s Disease.")
