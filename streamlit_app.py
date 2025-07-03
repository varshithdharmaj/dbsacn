import streamlit as st
import numpy as np
import pickle
import plotly.express as px

# Avoid naming this file as streamlit.py when running
# Load model bundle
with open("parkinson_model.pkl", "rb") as f:
    bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["features"]

# App setup
st.set_page_config(page_title="Parkinson's Detection", layout="centered")
st.title("🧠 Parkinson's Disease Detection App")
st.markdown("This app uses a machine learning model to detect **Parkinson's Disease** from biomedical voice features.")

# Sidebar info
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    - Enter the required voice features below.
    - Press **Predict** to see the result.
    """)
    st.caption("Developed by Varshith Dharmaj")

# Input form
st.subheader("Enter the values below:")
input_data = {}

with st.form("input_form"):
    col1, col2 = st.columns(2)
    for i, feature in enumerate(feature_names):
        with (col1 if i % 2 == 0 else col2):
            input_data[feature] = st.number_input(feature, step=0.001, format="%.6f")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        values = np.array([input_data[f] for f in feature_names]).reshape(1, -1)
        values_scaled = scaler.transform(values)
        prediction = model.predict(values_scaled)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(values_scaled)[0]
            confidence = max(prob) * 100
        else:
            prob = [0.5, 0.5]
            confidence = "N/A"

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("🔴 The person is likely to have **Parkinson's Disease**.")
        else:
            st.success("🟢 The person is **unlikely to have Parkinson's Disease**.")

        if isinstance(confidence, float):
            st.markdown(f"**Model Confidence:** {confidence:.2f}%")

        # Display bar chart
        fig = px.bar(
            x=["Parkinson's", "Healthy"],
            y=prob,
            labels={"x": "Class", "y": "Probability"},
            title="Prediction Probability"
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
