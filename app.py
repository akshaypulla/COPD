import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# Define the path to the output directory
OUTPUT_DIR = 'output_models'
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, 'preprocessor.pkl')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'final_selected_model.pkl')

# --- Load the preprocessor and model ---
@st.cache_resource # Cache the resource to avoid reloading on each rerun
def load_resources(preprocessor_path, model_path):
    """Loads the preprocessor and the trained model."""
    try:
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return preprocessor, model
    except FileNotFoundError:
        st.error(f"Error: Model or preprocessor file not found. Make sure '{OUTPUT_DIR}' directory exists and contains '{os.path.basename(preprocessor_path)}' and '{os.path.basename(model_path)}'.")
        return None, None
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

preprocessor, model = load_resources(PREPROCESSOR_PATH, MODEL_PATH)

# --- Streamlit UI ---
st.title("COPD Exacerbation Risk Prediction")

if preprocessor is not None and model is not None:
    st.write("Enter patient data to predict the risk of COPD exacerbation.")

    # Define input fields based on the features used in copd_analysis.py
    # Features identified from copd_analysis.py and CSV header (excluding patient_id, timestamp, exacerbation_risk):
    # SpO2, HRV, activity, sleep, PM2.5, NO2, pollen, humidity, cough, wheeze, FEV1, FVC, medications, comorbidities

    # Numerical features (based on copd_analysis.py)
    spo2 = st.number_input("SpO2 (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1)
    hrv = st.number_input("HRV (ms)", min_value=0.0, value=50.0, step=0.1)
    activity = st.number_input("Activity (steps/day)", min_value=0, value=1000, step=100)
    sleep = st.number_input("Sleep (hours/night)", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=20.0, step=0.1)
    no2 = st.number_input("NO2 (ppb)", min_value=0.0, value=15.0, step=0.1)
    pollen = st.number_input("Pollen Count", min_value=0.0, value=10.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
    fev1 = st.number_input("FEV1 (L)", min_value=0.0, value=2.0, step=0.01)
    fvc = st.number_input("FVC (L)", min_value=0.0, value=3.0, step=0.01)

    # Categorical features (based on copd_analysis.py)
    # Assuming these are numerical counts or categories based on the script's handling
    cough = st.number_input("Cough Score", min_value=0, value=0, step=1) # Assuming numerical score
    wheeze = st.number_input("Wheeze Score", min_value=0.0, value=0.0, step=0.1) # Assuming numerical score
    medications = st.number_input("Number of Medications", min_value=0, value=1, step=1) # Assuming numerical count
    comorbidities = st.number_input("Number of Comorbidities", min_value=0, value=1, step=1) # Assuming numerical count


    # Create a dictionary from the input values
    input_data = {
        'SpO2': spo2,
        'HRV': hrv,
        'activity': activity,
        'sleep': sleep,
        'PM2.5': pm25,
        'NO2': no2,
        'pollen': pollen,
        'humidity': humidity,
        'cough': cough,
        'wheeze': wheeze,
        'FEV1': fev1,
        'FVC': fvc,
        'medications': medications,
        'comorbidities': comorbidities
    }

    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Add a predict button
    if st.button("Predict Risk"):
        try:
            # Preprocess the input data
            # Need to ensure the columns are in the same order as during training
            # The preprocessor expects columns in the order it was fitted on.
            # We can get the feature names from the preprocessor if needed,
            # but for now, we rely on the order defined in input_data.
            # A safer approach would be to get feature names from the preprocessor
            # or the original training data columns.
            # Let's assume the order in input_data matches the training data for now.

            # The original script drops 'patient_id', 'timestamp', 'exacerbation_risk'
            # The preprocessor was fitted on the remaining columns.
            # We need to ensure the input_df has these columns in the correct order.
            # A robust way is to get the columns from the preprocessor's fitted state.
            # However, preprocessor.get_feature_names_out() gives names *after* encoding.
            # We need the names *before* preprocessing.
            # Let's read the original CSV header again to confirm the order of features used for X.
            # From the previous read_file, the header is:
            # patient_id,timestamp,SpO2,HRV,activity,sleep,PM2.5,NO2,pollen,humidity,cough,wheeze,FEV1,FVC,medications,comorbidities,exacerbation_risk
            # Features used for X are all except patient_id, timestamp, exacerbation_risk.
            # So the order should be:
            # SpO2,HRV,activity,sleep,PM2.5,NO2,pollen,humidity,cough,wheeze,FEV1,FVC,medications,comorbidities

            feature_order = ['SpO2', 'HRV', 'activity', 'sleep', 'PM2.5', 'NO2', 'pollen', 'humidity', 'cough', 'wheeze', 'FEV1', 'FVC', 'medications', 'comorbidities']
            input_df = input_df[feature_order] # Ensure correct column order

            input_processed = preprocessor.transform(input_df)

            # Make prediction
            prediction = model.predict(input_processed)
            prediction_proba = model.predict_proba(input_processed)

            st.subheader("Prediction Result:")
            risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
            st.write(f"The predicted COPD exacerbation risk is: **{risk_level}**")
            st.write(f"Probability of Low Risk: {prediction_proba[0][0]:.2f}")
            st.write(f"Probability of High Risk: {prediction_proba[0][1]:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Model and preprocessor could not be loaded. Please ensure the analysis script was run successfully.")

st.sidebar.header("About")
st.sidebar.info("This application predicts the risk of COPD exacerbation based on patient data.")