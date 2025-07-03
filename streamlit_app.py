import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- Load Models and Scalers ---
def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ File '{file}' not found.")
        return None

model_6 = load_pickle("model_6.pkl")
scaler_6 = load_pickle("scaler_6.pkl")

model_22 = load_pickle("model_22.pkl")
scaler_22 = load_pickle("scaler_22.pkl")

# --- Feature Sets ---
features_6 = ['fo', 'fhi', 'flo', 'jitter_percent', 'rap', 'ppe']
features_22 = [
    'mdvp:fo(Hz)', 'mdvp:fhi(Hz)', 'mdvp:flo(Hz)', 'mdvp:jitter(%)', 'mdvp:jitter(abs)', 'mdvp:rap', 'mdvp:ppq',
    'jitter:dda', 'mdvp:shimmer', 'mdvp:shimmer(db)', 'shimmer:apq3', 'shimmer:apq5', 'mdvp:apq',
    'shimmer:dda', 'nhr', 'hnr', 'rpde', 'dfa', 'spread1', 'spread2', 'd2', 'ppe'
]

# --- UI ---
st.title("🧠 Parkinson’s Disease Prediction App")
st.markdown("Upload data or enter manually. Choose 6 or 22 feature model.")

model_choice = st.selectbox("Select Model Type", ["6-feature", "22-feature"])

# --- Input Method ---
input_method = st.radio("Choose Input Method", ["📁 Upload CSV", "✏️ Manual Input", "🔣 Comma-Separated Input"])

df_input = None

if input_method == "📁 Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.success("✅ CSV loaded successfully!")

elif input_method == "🔣 Comma-Separated Input":
    comma_input = st.text_input("Enter comma-separated values:")
    if comma_input:
        values = [float(v.strip()) for v in comma_input.split(',')]
        selected_features = features_6 if model_choice == "6-feature" else features_22
        if len(values) != len(selected_features):
            st.error(f"❌ Expected {len(selected_features)} values, got {len(values)}.")
        else:
            df_input = pd.DataFrame([values], columns=selected_features)
            st.success("✅ Input parsed successfully!")

elif input_method == "✏️ Manual Input":
    selected_features = features_6 if model_choice == "6-feature" else features_22
    st.markdown("### Enter Feature Values Manually")
    values = []
    for feat in selected_features:
        val = st.number_input(feat, value=0.0)
        values.append(val)
    if st.button("Submit Manual Input"):
        df_input = pd.DataFrame([values], columns=selected_features)
        st.success("✅ Manual input captured!")

# --- Prediction ---
if df_input is not None:
    st.subheader("🔍 Input Preview")
    st.dataframe(df_input)

    st.subheader("📊 Input Visualization")
    st.bar_chart(df_input.T)

    selected_features = features_6 if model_choice == "6-feature" else features_22
    model = model_6 if model_choice == "6-feature" else model_22
    scaler = scaler_6 if model_choice == "6-feature" else scaler_22

    if model is None or scaler is None:
        st.error("❌ Model or scaler not loaded.")
    else:
        try:
            df_scaled = scaler.transform(df_input[selected_features])
            prediction = model.predict(df_scaled)[0]
            proba = model.predict_proba(df_scaled)[0][prediction]
            if prediction == 1:
                st.error(f"🔍 Prediction: Parkinson's Likely (Confidence: {proba:.2f})")
            else:
                st.success(f"✅ Prediction: Healthy (Confidence: {proba:.2f})")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
