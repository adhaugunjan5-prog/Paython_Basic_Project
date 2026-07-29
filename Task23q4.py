import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix


# Load Dataset

data = load_breast_cancer()

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


# Naive Bayes Model

nb = GaussianNB()

nb.fit(
    X_train,
    y_train
)


# Prediction

nb_pred = nb.predict(
    X_test
)


# Streamlit Output

st.title("Naive Bayes Classification")


st.subheader("Classification Report")

st.text(
    classification_report(
        y_test,
        nb_pred
    )
)


st.subheader("Confusion Matrix")

st.write(
    confusion_matrix(
        y_test,
        nb_pred
    )
)