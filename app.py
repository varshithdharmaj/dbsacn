import streamlit as st
import numpy as np
import pickle
import plotly.express as px

# Load the trained model
with open("parkinson_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define feature names in order
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:RAP', 'PPE',
    'Shimmer', 'APQ', 'DFA', 'Spread1', 'Spread2',
    'MDVP:Shimmer(dB)', 'HNR', 'MDVP:Jitter(Abs)', 'D2', 'DDP'
]

# Set up Streamlit page
st.set_page_config(page_title="Parkinson's Detection", layout="centered")
st.title("🧠 Parkinson's Disease Detection App")
st.markdown("This app uses machine learning to predict if a person has **Parkinson's Disease** based on voice measurements.")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    - **16 biomedical voice measurements**
    - Model: Trained using scikit-learn
    - Predicts if subject has Parkinson's
    """)
    st.caption("Created by Varshith Dharmaj")

# Input form
st.subheader("Enter the voice measurement values:")
input_data = {}

with st.form("input_form"):
    col1, col2 = st.columns(2)
    for i, feature in enumerate(feature_names):
        with (col1 if i % 2 == 0 else col2):
            input_data[feature] = st.number_input(
                feature, step=0.001, format="%.6f")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_values = np.array([input_data[f] for f in feature_names]).reshape(1, -1)
        prediction = model.predict(input_values)[0]

        # Optional: Get prediction probability (if model supports it)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_values)[0]
            confidence = max(probabilities) * 100
        else:
            confidence = None

        st.subheader("Result:")
        if prediction == 1:
            st.error("🔴 The model predicts that the person **has Parkinson's Disease**.")
        else:
            st.success("🟢 The model predicts that the person **does NOT have Parkinson's Disease**.")

        if confidence:
            st.markdown(f"**Prediction Confidence:** {confidence:.2f}%")

        # Display probability chart if available
        if confidence:
            fig = px.bar(
                x=["Parkinson's", "Healthy"],
                y=probabilities,
                labels={"x": "Class", "y": "Probability"},
                title="Prediction Probability"
            )
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
