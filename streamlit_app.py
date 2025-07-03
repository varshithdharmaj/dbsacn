import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import pytesseract
from PIL import Image
import plotly.graph_objects as go

# Optional: Set path if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

st.set_page_config(page_title="Parkinson's Predictor", layout="centered")
st.title("🧠 Parkinson’s Disease Prediction App")

# Load Models and Scalers Safely
def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ File '{file}' not found.")
        return None

# Load models and scalers
model_6 = load_pickle("model_6.pkl")
scaler_6 = load_pickle("scaler_6.pkl")
model_22 = load_pickle("model_22.pkl")
scaler_22 = load_pickle("scaler_22.pkl")

# Feature sets
features_6 = ['fo', 'fhi', 'flo', 'jitter_percent', 'rap', 'ppe']
features_22 = features_6 + [f'feature_{i}' for i in range(7, 23)]  # Customize as needed

# Model Selection
model_choice = st.radio("🔍 Select Model:", ["6-feature", "22-feature"])
selected_features = features_6 if model_choice == "6-feature" else features_22
model = model_6 if model_choice == "6-feature" else model_22
scaler = scaler_6 if model_choice == "6-feature" else scaler_22

# Input Method
input_mode = st.selectbox("🧾 Choose Input Method:", [
    "Upload CSV",
    "Manual Entry Form",
    "Comma-Separated Text",
    "Upload Image (OCR)" if model_choice == "6-feature" else "Upload Image (OCR) (Disabled)"
])

df_input = None

# --- CSV Upload ---
if input_mode == "Upload CSV":
    uploaded = st.file_uploader("📄 Upload CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df)

        if all(f in df.columns for f in selected_features):
            df_input = df[selected_features].iloc[-1:]
        else:
            st.error("❌ Required features not found in uploaded file.")

# --- Manual Entry Form ---
elif input_mode == "Manual Entry Form":
    st.info("Enter each value:")
    values = []
    for feat in selected_features:
        val = st.number_input(f"{feat}", key=feat)
        values.append(val)
    if st.button("Submit Manual Input"):
        df_input = pd.DataFrame([values], columns=selected_features)

# --- Comma-Separated Text Input ---
elif input_mode == "Comma-Separated Text":
    user_input = st.text_area("Paste values separated by commas:")
    if st.button("Submit Text Input"):
        try:
            values = list(map(float, user_input.strip().split(',')))
            if len(values) != len(selected_features):
                st.error(f"❌ Please provide exactly {len(selected_features)} values.")
            else:
                df_input = pd.DataFrame([values], columns=selected_features)
        except:
            st.error("⚠️ Please ensure all inputs are valid numbers.")

# --- Image (OCR) Input ---
elif input_mode.startswith("Upload Image") and model_choice == "6-feature":
    image = st.file_uploader("🖼️ Upload image (with 6 numeric values)", type=["png", "jpg", "jpeg"])
    if image:
        img = Image.open(image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        try:
            text = pytesseract.image_to_string(img)
            st.text_area("📋 Extracted Text", text, height=150)

            values = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            values = list(map(float, values))

            if len(values) >= 6:
                df_input = pd.DataFrame([values[:6]], columns=features_6)
            else:
                st.warning("❗ OCR did not extract enough values. Need at least 6.")
        except pytesseract.pytesseract.TesseractNotFoundError:
            st.error("❌ Tesseract-OCR not found. Please install it or set its path.")
        except Exception as e:
            st.error(f"❌ OCR failed: {str(e)}")

# --- Prediction Section ---
if df_input is not None:
    st.subheader("✅ Input Data Preview")
    st.dataframe(df_input)

    st.subheader("📊 Input Visualization")
    fig = go.Figure(data=[go.Bar(x=df_input.columns, y=df_input.iloc[0].values)])
    fig.update_layout(title="Feature Values")
    st.plotly_chart(fig)

    if model is None or scaler is None:
        st.error("❌ Model or scaler not loaded.")
    else:
        try:
            X_scaled = scaler.transform(df_input[selected_features])
            prediction = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0][prediction]

            if prediction == 1:
                st.error(f"🔍 Prediction: **Parkinson's Likely** (Confidence: {proba:.2f})")
            else:
                st.success(f"✅ Prediction: **Healthy** (Confidence: {proba:.2f})")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
