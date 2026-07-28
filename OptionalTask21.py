# ==============================
# House Price Prediction App
# ==============================

# Import Libraries
import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Model and Preprocessing Files
# ==============================
model = joblib.load("HousePriceModel.pkl")
scaler = joblib.load("HousePriceScaler.pkl")
encoded_columns = joblib.load("HousePriceColumns.pkl")

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

# ==============================
# Title
# ==============================
st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict the selling price.")

# ==============================
# Numerical Inputs
# ==============================
OverallQual = st.number_input(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=5
)

GrLivArea = st.number_input(
    "Ground Living Area (sq ft)",
    min_value=100,
    max_value=6000,
    value=1500
)

GarageArea = st.number_input(
    "Garage Area (sq ft)",
    min_value=0,
    max_value=1500,
    value=500
)

TotalBsmtSF = st.number_input(
    "Basement Area (sq ft)",
    min_value=0,
    max_value=7000,
    value=800
)

LotArea = st.number_input(
    "Lot Area (sq ft)",
    min_value=1000,
    max_value=250000,
    value=9000
)

# ==============================
# Categorical Inputs
# ==============================
Neighborhood = st.selectbox(
    "Neighborhood",
    ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"]
)

HouseStyle = st.selectbox(
    "House Style",
    ["1Story", "2Story", "1.5Fin"]
)

# ==============================
# Prediction Button
# ==============================
if st.button("Predict House Price"):

    try:
        # Create DataFrame
        input_data = pd.DataFrame({
            "OverallQual": [OverallQual],
            "GrLivArea": [GrLivArea],
            "GarageArea": [GarageArea],
            "TotalBsmtSF": [TotalBsmtSF],
            "LotArea": [LotArea],
            "Neighborhood": [Neighborhood],
            "HouseStyle": [HouseStyle]
        })

        # One-Hot Encoding
        input_encoded = pd.get_dummies(input_data)

        # Match training columns
        input_encoded = input_encoded.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        # Scale Numerical Features
        numerical_columns = [
            "OverallQual",
            "GrLivArea",
            "GarageArea",
            "TotalBsmtSF",
            "LotArea"
        ]

        input_encoded[numerical_columns] = scaler.transform(
            input_encoded[numerical_columns]
        )

        # Prediction
        prediction = model.predict(input_encoded)

        # Display Result
        st.success(
            f"🏡 Predicted House Price: ${prediction[0]:,.2f}"
        )

    except Exception as e:
        st.error(f"Error: {e}")