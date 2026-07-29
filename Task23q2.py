# Q2 Logistic Regression

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)



data = load_breast_cancer()


X = pd.DataFrame(
data.data,
columns=data.feature_names
)

y = data.target



X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.2,
random_state=42
)


scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)



model=LogisticRegression()

model.fit(
X_train,
y_train
)


y_pred=model.predict(
X_test
)


import streamlit as st

st.title("Logistic Regression Results")

st.write("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))