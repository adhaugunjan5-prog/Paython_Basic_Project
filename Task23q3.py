import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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


# KNN

accuracy = []

for k in [3,5,7]:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(
        X_train,
        y_train
    )

    pred = knn.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        pred
    )

    accuracy.append(
        [k, acc]
    )


result = pd.DataFrame(
    accuracy,
    columns=[
        "K Value",
        "Accuracy"
    ]
)


# Streamlit Output

st.title("KNN Classification")

st.dataframe(result)


best_k = result.loc[
    result["Accuracy"].idxmax()
]


st.write("Best K:")
st.write(best_k)