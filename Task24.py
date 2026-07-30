# ==========================================
# MACHINE LEARNING ASSIGNMENT
# ==========================================

import streamlit as st
import pandas as pd

from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ------------------------------------------
# Streamlit Page
# ------------------------------------------

st.set_page_config(
    page_title="Machine Learning Assignment",
    layout="wide"
)

st.title("Machine Learning Assignment")
st.write("Select any question from the sidebar.")

# ------------------------------------------
# Sidebar
# ------------------------------------------

option = st.sidebar.radio(
    "Select Question",
    (
        "Q1 Linear Regression",
        "Q2 Logistic Regression",
        "Q3 KNN",
        "Q4 Naive Bayes",
        "Q5 Algorithm Comparison"
    )
)

# ==========================================
# Q1 : Linear Regression
# ==========================================

if option == "Q1 Linear Regression":

    st.header("Q1 : Linear Regression")

    # Load Dataset
    housing = fetch_california_housing()

    X = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    y = housing.target

    st.subheader("First 5 Rows")
    st.dataframe(X.head())

    st.write("Dataset Shape :", X.shape)

    st.subheader("Missing Values")
    st.write(X.isnull().sum())

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = LinearRegression()

    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    r2 = r2_score(y_test, y_pred)

    st.subheader("R² Score")

    st.success(round(r2,4))

    prediction_df = pd.DataFrame({

        "Actual": y_test[:10],

        "Predicted": y_pred[:10]

    })

    st.subheader("First 10 Predictions")

    st.dataframe(prediction_df)

    # ==========================================
# Q2 : Logistic Regression
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

if option == "Q2 Logistic Regression":

    st.header("Q2 : Logistic Regression")

    cancer = load_breast_cancer()

    X = pd.DataFrame(
        cancer.data,
        columns=cancer.feature_names
    )

    y = cancer.target

    st.subheader("First 5 Rows")
    st.dataframe(X.head())

    st.write("Dataset Shape :", X.shape)

    st.subheader("Missing Values")
    st.write(X.isnull().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("Model Performance")

    st.write("Accuracy :", round(accuracy,4))
    st.write("Precision :", round(precision,4))
    st.write("Recall :", round(recall,4))
    st.write("F1 Score :", round(f1,4))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test,y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y_test,y_pred))


# ==========================================
# Q3 : K-Nearest Neighbors
# ==========================================

from sklearn.neighbors import KNeighborsClassifier

if option == "Q3 KNN":

    st.header("Q3 : K-Nearest Neighbors")

    cancer = load_breast_cancer()

    X = pd.DataFrame(
        cancer.data,
        columns=cancer.feature_names
    )

    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []

    best_accuracy = 0
    best_k = 0

    for k in [3,5,7]:

        model = KNeighborsClassifier(
            n_neighbors=k
        )

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        results.append({
            "K Value":k,
            "Accuracy":round(accuracy,4)
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    comparison_df = pd.DataFrame(results)

    st.subheader("Comparison Table")

    st.dataframe(comparison_df)

    st.success(
        f"Best K = {best_k}"
    )

    st.write(
        "Best Accuracy :",
        round(best_accuracy,4)
    )

    # ==========================================
# Q4 : Naive Bayes
# ==========================================

from sklearn.naive_bayes import GaussianNB

if option == "Q4 Naive Bayes":

    st.header("Q4 : Naive Bayes")

    cancer = load_breast_cancer()

    X = pd.DataFrame(
        cancer.data,
        columns=cancer.feature_names
    )

    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GaussianNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Accuracy")
    st.success(round(accuracy,4))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))


# ==========================================
# Q5 : Algorithm Comparison
# ==========================================

if option == "Q5 Algorithm Comparison":

    st.header("Q5 : Algorithm Comparison")

    cancer = load_breast_cancer()

    X = pd.DataFrame(
        cancer.data,
        columns=cancer.feature_names
    )

    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Algorithm": name,
            "Accuracy": round(accuracy,4),
            "Precision": round(precision,4),
            "Recall": round(recall,4),
            "F1 Score": round(f1,4)
        })

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    st.subheader("Comparison Table")
    st.dataframe(comparison_df)

    best_algorithm = comparison_df.iloc[0]["Algorithm"]
    best_accuracy = comparison_df.iloc[0]["Accuracy"]

    st.success(f"Best Algorithm : {best_algorithm}")

    st.write("Accuracy :", best_accuracy)

    st.info(
        f"{best_algorithm} achieved the highest accuracy among all algorithms."
    )