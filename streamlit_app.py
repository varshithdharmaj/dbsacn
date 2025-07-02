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

# Load scaler if available
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as s:
        scaler = pickle.load(s)
    scaler_loaded = True
else:
    scaler_loaded = False

# Features used during training
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
    'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]
import io

# Sample data row
sample_row = [119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370, 0.00554, 0.01109,
              0.04374, 0.426, 0.02182, 0.03130, 0.02971, 0.06545, 0.02211,
              21.033, 0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654]

# Convert to CSV for download
csv_data = pd.DataFrame([sample_row], columns=feature_names).to_csv(index=False)
st.download_button("📥 Download Sample CSV", csv_data, file_name="sample_input.csv", mime="text/csv")


# UI setup
st.set_page_config(page_title="Parkinson's Prediction", layout="centered")
st.title("🧠 Parkinson’s Disease Prediction")
st.markdown("Choose how you want to input the data:")

# Input method
input_mode = st.radio("Select Input Method", ["Manual Entry", "CSV Upload", "Paste Values", "Upload Screenshot (Coming Soon)"])


input_data = None

if input_mode == "Manual Entry":
    st.markdown("Enter values for each feature:")
    input_features = []
    for name in feature_names:
        value = st.number_input(name, step=0.001, format="%.6f")
        input_features.append(value)
    input_data = np.array([input_features])

elif input_mode == "CSV Upload":
    uploaded_file = st.file_uploader("Upload a CSV file (1 row with 22 columns)", type=["csv"])
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
    text_input = st.text_area("Paste 22 comma-separated values below:", height=100)
    if text_input:
        try:
            values = [float(x.strip()) for x in text_input.split(",")]
            if len(values) != 22:
                st.error("Exactly 22 values are required.")
            else:
                input_data = np.array([values])
        except ValueError:
            st.error("Ensure all values are numeric and separated by commas.")
elif input_mode == "Upload Screenshot (Coming Soon)":
    st.info("🖼️ Upload a screenshot of your report. This feature will auto-extract values in future updates.")
    uploaded_image = st.file_uploader("Upload Image (PNG or JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Screenshot", use_column_width=True)
        st.warning("🔧 OCR-based auto-detection coming soon! For now, use another method to enter values.")


# Predict
if input_data is not None and st.button("Predict"):
    if scaler_loaded:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Likely Parkinson's Disease Detected.\n**Probability:** {probability:.2%}")
    else:
        st.success(f"✅ Unlikely to have Parkinson's Disease.\n**Probability:** {probability:.2%}")
