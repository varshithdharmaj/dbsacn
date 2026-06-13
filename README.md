

🧠. Parkinson's Disease Detection 

This project detects the presence of Parkinson’s Disease using biomedical voice measurements. It uses supervised ML models trained on the UCI Parkinson’s dataset to classify patients based on extracted features from vocal recordings.
  # Parkinson's Disease Detection using ML

This project uses machine learning models to detect Parkinson's disease based on vocal measurements from UCI’s dataset.

https://parkinsonusingml-dgknf8x8r5gkjd52zuaudu.streamlit.app/

## 📁 Dataset
- Source: UCI Machine Learning Repository
- Features include jitter, shimmer, and fundamental frequency
- Target: `status` (1 = Parkinson’s, 0 = Healthy)

## 🧠 Models Used
- Logistic Regression
- Random Forest
- SHAP Explainability

## 📈 Results
- Accuracy: ~X%
- AUC-ROC:  ~X

## 🚀 Future Work
- Voice-based deep learning
- Streamlit UI
- Cloud deployment (GCP/Render)

## ✅ How to Run
```bash
pip install -r requirements.txt
