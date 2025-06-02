import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

st.title("?? QSAR Alzheimer's Predictor")

# --- Download Regressor Model from Google Drive ---
regressor_file = "alz_qsar_admet_bbb_model_compressed.pkl"
if not os.path.exists(regressor_file):
    url = "https://drive.google.com/uc?id=11pGE9HLVbd9dDd8GWKG6ZQIKztFjlAQA"
    gdown.download(url, regressor_file, quiet=False)

# --- Load Models ---
classifier = joblib.load("alz_qsar_classifier_compressed.pkl")  # stored locally in repo
regressor = joblib.load(regressor_file)

# --- Upload CSV ---
uploaded_file = st.file_uploader("?? Upload ECFP4 Fingerprints CSV (Bit_0 to Bit_1023)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check fingerprint columns
    bit_columns = [col for col in df.columns if col.startswith("Bit_")]
    if len(bit_columns) != 1024:
        st.error("? Expected 1024 fingerprint bits (Bit_0 to Bit_1023).")
    else:
        X = df[bit_columns].values

        # Make predictions
        st.subheader("?? Predictions")
        df["Alzheimer's_Prob"] = classifier.predict_proba(X)[:, 1]
        df["Predicted_pIC50"] = regressor.predict(X)

        st.success("? Prediction complete!")
        st.dataframe(df[["Alzheimer's_Prob", "Predicted_pIC50"] + (["SMILES"] if "SMILES" in df.columns else [])])

        # Downloadable CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("?? Download Results CSV", data=csv, file_name="alz_qsar_predictions.csv", mime="text/csv")
