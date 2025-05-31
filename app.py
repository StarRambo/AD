import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

st.title("QSAR Model: pIC50 Predictor for Alzheimer's (Classification + Regression)")

# Define paths and Google Drive IDs
classifier_path = "alz_qsar_classifier.pkl"
classifier_url = "https://drive.google.com/uc?id=14W54IDjC6pqjkmDj44zQmQrRdXm4ViwB"

regressor_path = "alz_qsar_admet_bbb_model_compressed.pkl"
regressor_url = "https://drive.google.com/uc?id=1cOcNBhnyhq0QfBrdDCRgghvHlMI4sgcd"

# Download if not already available
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
        df["Class_Prediction"] = clf.predict(X)
        df["Predicted_pIC50"] = reg.predict(X)

        st.success("Prediction complete!")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "predicted_output.csv", "text/csv")
