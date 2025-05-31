import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

st.title("QSAR Model: pIC50 Predictor for Alzheimer's (with ADMET & BBB)")

# Compressed model path and download link
model_path = "alz_qsar_admet_bbb_model_compressed.pkl"
model_url = "https://drive.google.com/uc?id=1cOcNBhnyhq0QfBrdDCRgghvHlMI4sgcd"

# Download model if not present
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load the compressed model
model = joblib.load(model_path)

uploaded_file = st.file_uploader("Upload CSV with 1022-bit ECFP4 fingerprints", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    bit_columns = [col for col in df.columns if col.startswith("Bit_")]

    if len(bit_columns) != 1022:
        st.error("Expected 1022 fingerprint bits (Bit_0 to Bit_1021).")
    else:
        X = df[bit_columns].values
        predictions = model.predict(X)
        df["Predicted_Output"] = predictions
        st.success("Prediction complete!")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predicted_output.csv", "text/csv")
