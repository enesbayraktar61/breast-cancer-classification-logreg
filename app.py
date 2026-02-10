import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

# Page config
st.set_page_config(page_title="Breast Cancer Classification", layout="centered")

st.title("Breast Cancer Classification")
st.write("Predict whether a tumor is **malignant** or **benign** using a Logistic Regression model.")

# Base directory (Hugging Face repo root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "breast_cancer_logreg_pipeline.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "training_columns.json")

# Load model pipeline
model = joblib.load(MODEL_PATH)

# Load training columns
with open(COLUMNS_PATH, "r") as f:
    training_columns = json.load(f)

# Build default values from sklearn dataset medians
data = load_breast_cancer()
defaults_df = pd.DataFrame(data.data, columns=data.feature_names)
default_values = defaults_df.median().to_dict()

st.subheader("Input Features")

st.info(
    "Defaults are set to dataset median values. You can adjust any feature and click **Predict**."
)

# Create inputs dynamically
user_input = {}
with st.expander("Show / Edit all features", expanded=False):
    for col in training_columns:
        default_val = float(default_values.get(col, 0.0))
        user_input[col] = st.number_input(
            label=col,
            value=default_val,
            format="%.6f"
        )

# Create DataFrame in correct column order
input_df = pd.DataFrame([user_input], columns=training_columns)

if st.button("Predict"):
    # Predict class (0 = malignant, 1 = benign)
    pred_class = int(model.predict(input_df)[0])

    # Predict probability if available
    proba_text = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        proba_malignant = float(proba[0])
        proba_benign = float(proba[1])
        proba_text = f"\n\n**Probability (Malignant):** {proba_malignant:.3f}  \n**Probability (Benign):** {proba_benign:.3f}"

    if pred_class == 0:
        st.error("Prediction: **Malignant (0)**" + proba_text)
    else:
        st.success("Prediction: **Benign (1)**" + proba_text)

st.caption("Model: Logistic Regression (StandardScaler + LogisticRegression) saved as a single sklearn Pipeline.")
st.caption("Disclaimer: This app is for educational purposes only and not for medical use.")
