# ==========================
# Q1. Setup and Libraries
# ==========================

# Streamlit is used to create the web application.
import streamlit as st

# Pandas is used to create and manipulate DataFrames.
import pandas as pd

# Joblib is used to load the trained model and preprocessing files.
import joblib


# ==========================
# Q3. Page Configuration
# ==========================

# Configure the Streamlit page for a better user interface.
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)


# ==========================
# Q2. Load Model Files
# ==========================

try:
    # Load the trained machine learning model.
    model = joblib.load("LR_ford_car.pkl")

    # Load the StandardScaler.
    scaler = joblib.load("scaler.pkl")

    # Load the encoded column names.
    encoded_columns = joblib.load("columns.pkl")

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


# ==========================
# Q4. Title and Description
# ==========================

st.title("🚗 Ford Car Price Predictor")

st.write(
    "Enter the car details below to predict its selling price."
)


# ==========================
# Q5. Numerical Input Fields
# ==========================

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2018
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=300000,
    value=50000
)

tax = st.number_input(
    "Road Tax",
    min_value=0,
    max_value=1000,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    max_value=150.0,
    value=50.0
)

engineSize = st.number_input(
    "Engine Size",
    min_value=0.0,
    max_value=6.0,
    value=1.5
)


# ==========================
# Q6. Dropdown Inputs
# ==========================

# Selectbox prevents invalid user input.
transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual", "Semi-Auto"]
)

# Selectbox provides only valid fuel type options.
fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)


# ==========================
# Q7. Text Input & Button
# ==========================

model_name = st.text_input("Car Model")

predict = st.button("Predict Price")


# ==========================
# Q8 & Q9
# ==========================

if predict:

    try:

        # Create DataFrame from user inputs.
        input_data = pd.DataFrame({
            "model": [model_name],
            "year": [year],
            "transmission": [transmission],
            "mileage": [mileage],
            "fuelType": [fuelType],
            "tax": [tax],
            "mpg": [mpg],
            "engineSize": [engineSize]
        })

        # Apply One-Hot Encoding.
        input_encoded = pd.get_dummies(input_data)

        # Match training columns.
        input_encoded = input_encoded.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        # Numerical columns.
        numerical_columns = [
            "year",
            "mileage",
            "tax",
            "mpg",
            "engineSize"
        ]

        # Apply feature scaling.
        input_encoded[numerical_columns] = scaler.transform(
            input_encoded[numerical_columns]
        )

        # Predict price.
        prediction = model.predict(input_encoded)

        # Display result.
        st.success(
            f"Predicted Car Price: £ {prediction[0]:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")