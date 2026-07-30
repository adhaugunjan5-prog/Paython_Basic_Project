import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Page Configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="centered"
)


st.title("📚 Student Performance Prediction Model")

st.write(
    "Machine Learning model to predict student's final exam score (G3)"
)


# Load Dataset

@st.cache_data
def load_data():
    df = pd.read_csv("student-mat.csv")
    return df


df = load_data()


# Dataset Preview

st.subheader("Dataset Preview")
st.dataframe(df.head())


# Dataset Information

st.subheader("Dataset Shape")
st.write(df.shape)


st.subheader("Missing Values")
st.write(df.isnull().sum())


# Encoding

le = LabelEncoder()

for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col])


# Features and Target

X = df.drop("G3", axis=1)

y = df["G3"]


# Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# Train Models

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)



dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train,y_train)

dt_pred = dt.predict(X_test)



rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train,y_train)

rf_pred = rf.predict(X_test)



# Model Results

results = pd.DataFrame({

    "Model":[
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "R2 Score":[
        r2_score(y_test,lr_pred),
        r2_score(y_test,dt_pred),
        r2_score(y_test,rf_pred)
    ]

})


st.subheader("📊 Model Comparison")

st.dataframe(results)



# Best Model

best_model = results.loc[
    results["R2 Score"].idxmax(),
    "Model"
]


st.success(
    f"Best Performing Model: {best_model}"
)



# Prediction Section

st.subheader("Prediction Example")


sample = X_test[0].reshape(1,-1)


prediction = rf.predict(sample)


st.write(
    "Predicted Student Score:",
    round(prediction[0],2)
)



# Metrics

st.subheader("Random Forest Evaluation")

st.write(
    "R2 Score:",
    r2_score(y_test,rf_pred)
)

st.write(
    "MAE:",
    mean_absolute_error(y_test,rf_pred)
)

st.write(
    "MSE:",
    mean_squared_error(y_test,rf_pred)
)