import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

st.title("QSAR Model: pIC50 Predictor for Alzheimer's")

# Define model file paths and Google Drive links
classifier_path = "rf_classifier.pkl"
regressor_path = "rf_regressor.pkl"

classifier_url = "https://drive.google.com/uc?id=14W54IDjC6pqjkmDj44zQmQrRdXm4ViwB"
regressor_url = "https://drive.google.com/uc?id=10uVXA4a4sPiablu_CJPdAlATHg4tf8k6"

# Download models if not present
if not os.path.exists(classifier_path):
    gdown.download(classifier_url, classifier_path, quiet=False)

if not os.path.exists(regressor_path):
    gdown.download(regressor_url, regressor_path, quiet=False)

# Load models
clf = joblib.load(classifier_path)
reg = joblib.load(regressor_path)

uploaded_file = st.file_uploader("Upload CSV with 1022-bit ECFP4 fingerprints", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    bit_columns = [col for col in df.columns if col.startswith("Bit_")]

    if len(bit_columns) != 1022:
        st.error("Expected 1022 fingerprint bits (Bit_0 to Bit_1021).")
    else:
        X = df[bit_columns].values
        pred_class = clf.predict(X)
        pred_pIC50 = reg.predict(X)
        df["Class_Prediction"] = pred_class
        df["Predicted_pIC50"] = pred_pIC50
        st.success("Prediction complete!")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predicted_output.csv", "text/csv")
