# Question No. 10

import streamlit as st
import pandas as pd
import joblib

# Load the saved model and column names
model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

# Title
st.title("❤️ Heart Disease Prediction")

# User Input Fields
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100
)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

cp = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

bp = st.number_input(
    "Resting BP"
)

chol = st.number_input(
    "Cholesterol"
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

hr = st.number_input(
    "Maximum Heart Rate"
)

angina = st.selectbox(
    "Exercise Angina",
    ["Y", "N"]
)

oldpeak = st.number_input(
    "Old Peak"
)

slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# Prediction Button
if st.button("Predict"):

    # Create a DataFrame from user input
    sample = {
        "Age": age,
        "RestingBP": bp,
        "Cholesterol": chol,
        "FastingBS": fbs,
        "MaxHR": hr,
        "Oldpeak": oldpeak,
        "Sex": sex,
        "ChestPainType": cp,
        "RestingECG": ecg,
        "ExerciseAngina": angina,
        "ST_Slope": slope
    }

    sample = pd.DataFrame([sample])

    # Apply One-Hot Encoding
    sample = pd.get_dummies(sample)

    # Match columns with training data
    sample = sample.reindex(
        columns=columns,
        fill_value=0
    )

    # Predict
    prediction = model.predict(sample)

    # Display Result
    if prediction[0] == 1:
        st.error("⚠️ Heart Disease: YES")
    else:
        st.success("✅ Heart Disease: NO")