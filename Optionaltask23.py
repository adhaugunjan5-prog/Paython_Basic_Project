import streamlit as st

st.sidebar.title("Machine Learning Assignment")

model_option = st.sidebar.selectbox(
    "Select Model",
    [
        "Linear Regression",
        "Logistic Regression",
        "KNN",
        "Naive Bayes"
    ]
)

# -----------------------------
# Linear Regression
# -----------------------------
if model_option == "Linear Regression":

    st.header("Linear Regression Prediction")

    MedInc = st.number_input("Median Income")
    HouseAge = st.number_input("House Age")
    AveRooms = st.number_input("Average Rooms")
    AveBedrms = st.number_input("Average Bedrooms")
    Population = st.number_input("Population")
    AveOccup = st.number_input("Average Occupancy")
    Latitude = st.number_input("Latitude")
    Longitude = st.number_input("Longitude")

    if st.button("Predict House Price"):
        st.success("Prediction Completed Successfully")

# -----------------------------
# Logistic Regression
# -----------------------------
elif model_option == "Logistic Regression":

    st.header("Breast Cancer Prediction")

    radius = st.number_input("Mean Radius")
    texture = st.number_input("Mean Texture")
    perimeter = st.number_input("Mean Perimeter")

    if st.button("Predict"):
        st.success("Prediction Completed Successfully")

# -----------------------------
# KNN
# -----------------------------
elif model_option == "KNN":

    st.header("KNN Prediction")

    radius = st.number_input("Mean Radius")
    texture = st.number_input("Mean Texture")

    if st.button("Predict using KNN"):
        st.success("Prediction Completed Successfully")

# -----------------------------
# Naive Bayes
# -----------------------------
elif model_option == "Naive Bayes":

    st.header("Naive Bayes Prediction")

    radius = st.number_input("Mean Radius")
    texture = st.number_input("Mean Texture")

    if st.button("Predict using Naive Bayes"):
        st.success("Prediction Completed Successfully")