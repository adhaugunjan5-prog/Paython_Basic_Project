# Q1 Linear Regression

import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Load Dataset

data = fetch_california_housing()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target


# Train Test Split

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



# Linear Regression Model

lr = LinearRegression()

lr.fit(
    X_train,
    y_train
)


# Prediction

prediction = lr.predict(
    X_test
)


# Evaluation

score = r2_score(
    y_test,
    prediction
)


import streamlit as st

st.title("Linear Regression")

st.write("## R² Score")
st.write(score)

st.write("## First 10 Predictions")
st.write(prediction[:10])