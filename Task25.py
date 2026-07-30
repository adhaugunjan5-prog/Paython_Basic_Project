import streamlit as st

st.set_page_config(
    page_title="ML Assignment",
    layout="wide"
)

st.sidebar.title("Select Question")

question = st.sidebar.selectbox(
    "Choose Question",
    [
        "Q1 - KNN Manual Tuning",
        "Q2 - SVM Manual Tuning",
        "Q3 - GridSearchCV",
        "Q4 - RandomizedSearchCV",
        "Q5 - Random Forest",
        "Q6 - AdaBoost & Gradient Boosting",
        "Q7 - XGBoost",
        "Q8 - Random Forest GridSearch",
        "Q9 - Model Comparison",
        "Q10 - Mini Project"
    ]
)

# ==========================================
# Q1 : Manual Hyperparameter Tuning - KNN
# ==========================================

import streamlit as st
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------
# Page Title
# ----------------------------

st.set_page_config(
    page_title="Q1 - KNN Hyperparameter Tuning",
    layout="wide"
)

st.title("Q1 : Manual Hyperparameter Tuning - KNN")

# ----------------------------
# Load Dataset
# ----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# ----------------------------
# Show Dataset
# ----------------------------

st.subheader("Dataset")

st.dataframe(X.head())

st.write("Dataset Shape :", X.shape)

st.write("Missing Values")

st.write(X.isnull().sum())

# ----------------------------
# Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

# ----------------------------
# Hyperparameter Tuning
# ----------------------------

k_values = [3, 5, 7, 11, 13, 15]

results = []

best_accuracy = 0
best_k = 0

st.subheader("KNN Accuracy")

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(X_train, y_train)

    accuracy = model.score(
        X_test,
        y_test
    )

    st.write(f"K = {k}  ➜ Accuracy = {accuracy:.4f}")

    results.append({

        "K Value": k,

        "Accuracy": round(
            accuracy,
            4
        )

    })

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_k = k

# ----------------------------
# Comparison Table
# ----------------------------

comparison_df = pd.DataFrame(results)

st.subheader("Comparison Table")

st.dataframe(comparison_df)

# ----------------------------
# Best Model
# ----------------------------

st.success(
    f"Best K Value : {best_k}"
)

st.success(
    f"Highest Accuracy : {best_accuracy:.4f}"
)

# ==========================================
# Q2 : Manual Hyperparameter Tuning - SVM
# ==========================================

from sklearn.svm import SVC

st.markdown("---")
st.title("Q2 : Manual Hyperparameter Tuning - SVM")

# ----------------------------
# Load Iris Dataset
# ----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# ----------------------------
# Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

# ----------------------------
# Hyperparameters
# ----------------------------

c_values = [1, 10, 20]
kernels = ["linear", "rbf"]

results = []

best_accuracy = 0
best_c = 0
best_kernel = ""

st.subheader("SVM Accuracy for Different Hyperparameters")

for c in c_values:

    for kernel in kernels:

        model = SVC(
            C=c,
            kernel=kernel,
            random_state=42
        )

        model.fit(X_train, y_train)

        accuracy = model.score(
            X_test,
            y_test
        )

        st.write(
            f"C = {c} | Kernel = {kernel} | Accuracy = {accuracy:.4f}"
        )

        results.append({

            "C": c,
            "Kernel": kernel,
            "Accuracy": round(accuracy,4)

        })

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_c = c
            best_kernel = kernel

# ----------------------------
# Comparison Table
# ----------------------------

comparison_df = pd.DataFrame(results)

st.subheader("Comparison Table")

st.dataframe(comparison_df)

# ----------------------------
# Best Hyperparameters
# ----------------------------

st.success(
    f"Best C Value : {best_c}"
)

st.success(
    f"Best Kernel : {best_kernel}"
)

st.success(
    f"Highest Accuracy : {best_accuracy:.4f}"
)

# ==========================================
# Q3 : GridSearchCV
# ==========================================

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

st.markdown("---")
st.title("Q3 : GridSearchCV")

# -----------------------------
# Load Iris Dataset
# -----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

# -----------------------------
# SVM Model
# -----------------------------

model = SVC()

# -----------------------------
# Parameter Grid
# -----------------------------

param_grid = {

    "C": [1, 10, 20],

    "kernel": ["linear", "rbf"]

}

# -----------------------------
# Grid Search
# -----------------------------

grid = GridSearchCV(

    estimator=model,

    param_grid=param_grid,

    cv=5,

    scoring="accuracy"

)

grid.fit(X_train, y_train)

# -----------------------------
# Results DataFrame
# -----------------------------

results = pd.DataFrame(grid.cv_results_)

results = results[
    [
        "param_C",
        "param_kernel",
        "mean_test_score"
    ]
]

st.subheader("Grid Search Results")

st.dataframe(results)

# -----------------------------
# Best Parameters
# -----------------------------

st.success(
    f"Best Parameters : {grid.best_params_}"
)

st.success(
    f"Best Score : {grid.best_score_:.4f}"
)

# ==========================================
# Q4 : Randomized Search CV
# ==========================================

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

st.markdown("---")
st.title("Q4 : Randomized Search CV")

# -----------------------------
# Load Iris Dataset
# -----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

# -----------------------------
# SVM Model
# -----------------------------

model = SVC()

# -----------------------------
# Parameter Grid
# -----------------------------

param_grid = {

    "C": [1, 10, 20],

    "kernel": ["linear", "rbf"]

}

# -----------------------------
# Randomized Search CV
# -----------------------------

random_search = RandomizedSearchCV(

    estimator=model,

    param_distributions=param_grid,

    n_iter=5,

    cv=5,

    random_state=42,

    scoring="accuracy"

)

random_search.fit(X_train, y_train)

# -----------------------------
# Results DataFrame
# -----------------------------

results = pd.DataFrame(
    random_search.cv_results_
)

results = results[
    [
        "param_C",
        "param_kernel",
        "mean_test_score"
    ]
]

st.subheader("Random Search Results")

st.dataframe(results)

# -----------------------------
# Best Parameters
# -----------------------------

st.success(
    f"Best Parameters : {random_search.best_params_}"
)

st.success(
    f"Best Score : {random_search.best_score_:.4f}"
)

# -----------------------------
# Grid Search vs Random Search
# -----------------------------

st.subheader("Comparison")

comparison = pd.DataFrame({

    "Method": [
        "Grid Search",
        "Random Search"
    ],

    "Best Score": [
        grid.best_score_,
        random_search.best_score_
    ]

})

st.dataframe(comparison)


# ==========================================
# Q5 : Bagging - Random Forest
# ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.markdown("---")
st.title("Q5 : Bagging - Random Forest")

# -----------------------------
# Load Iris Dataset
# -----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

st.subheader("Dataset")

st.dataframe(X.head())

st.write("Dataset Shape :", X.shape)

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# Random Forest Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.subheader("Model Accuracy")

st.success(f"Accuracy : {accuracy:.4f}")

# -----------------------------
# Actual vs Predicted
# -----------------------------

result_df = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_pred

})

st.subheader("Prediction Results")

st.dataframe(result_df.head(10))


# ==========================================
# Q6 : Boosting (AdaBoost & Gradient Boosting)
# ==========================================

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

st.markdown("---")
st.title("Q6 : Boosting - AdaBoost & Gradient Boosting")

# -----------------------------
# Load Iris Dataset
# -----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

st.subheader("Dataset")

st.dataframe(X.head())

st.write("Dataset Shape :", X.shape)

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# AdaBoost Model
# -----------------------------

ada_model = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

ada_model.fit(X_train, y_train)

ada_pred = ada_model.predict(X_test)

ada_accuracy = accuracy_score(
    y_test,
    ada_pred
)

# -----------------------------
# Gradient Boosting Model
# -----------------------------

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

gb_accuracy = accuracy_score(
    y_test,
    gb_pred
)

# -----------------------------
# Display Results
# -----------------------------

st.subheader("Accuracy Scores")

result_df = pd.DataFrame({

    "Model": [
        "AdaBoost",
        "Gradient Boosting"
    ],

    "Accuracy": [
        round(ada_accuracy,4),
        round(gb_accuracy,4)
    ]

})

st.dataframe(result_df)

# -----------------------------
# Best Model
# -----------------------------

if ada_accuracy > gb_accuracy:

    st.success(
        f"Best Model : AdaBoost ({ada_accuracy:.4f})"
    )

elif gb_accuracy > ada_accuracy:

    st.success(
        f"Best Model : Gradient Boosting ({gb_accuracy:.4f})"
    )

else:

    st.success(
        f"Both models achieved the same Accuracy ({ada_accuracy:.4f})"
    )

    # ==========================================
# Q7 : Boosting - XGBoost
# ==========================================

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.markdown("---")
st.title("Q7 : Boosting - XGBoost")

# -----------------------------
# Load Iris Dataset
# -----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

st.subheader("Dataset")

st.dataframe(X.head())

st.write("Dataset Shape :", X.shape)

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# XGBoost Model
# -----------------------------

model = XGBClassifier(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=3,

    random_state=42,

    use_label_encoder=False,

    eval_metric="mlogloss"

)

# -----------------------------
# Train Model
# -----------------------------

model.fit(
    X_train,
    y_train
)

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(
    X_test
)

# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.subheader("XGBoost Accuracy")

st.success(
    f"Accuracy : {accuracy:.4f}"
)

# -----------------------------
# Actual vs Predicted
# -----------------------------

result_df = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_pred

})

st.subheader("Prediction Results")

st.dataframe(result_df.head(10))


# ==========================================
# Q8 : Hyperparameter Tuning on Random Forest
# ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

st.markdown("---")
st.title("Q8 : Hyperparameter Tuning on Random Forest")

# -----------------------------
# Load Iris Dataset
# -----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

st.subheader("Dataset")

st.dataframe(X.head())

st.write("Dataset Shape :", X.shape)

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Random Forest Model
# -----------------------------

rf = RandomForestClassifier(
    random_state=42
)

# -----------------------------
# Parameter Grid
# -----------------------------

param_grid = {

    "n_estimators": [50, 100, 150],

    "max_depth": [3, 5, 7]

}

# -----------------------------
# Grid Search CV
# -----------------------------

grid = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=5,

    scoring="accuracy"

)

grid.fit(X_train, y_train)

# -----------------------------
# Results
# -----------------------------

results = pd.DataFrame(grid.cv_results_)

results = results[
    [
        "param_n_estimators",
        "param_max_depth",
        "mean_test_score"
    ]
]

st.subheader("Grid Search Results")

st.dataframe(results)

# -----------------------------
# Best Parameters
# -----------------------------

st.success(
    f"Best Parameters : {grid.best_params_}"
)

st.success(
    f"Best Score : {grid.best_score_:.4f}"
)

# -----------------------------
# Test Accuracy
# -----------------------------

best_model = grid.best_estimator_

accuracy = best_model.score(
    X_test,
    y_test
)

st.subheader("Test Accuracy")

st.success(
    f"Accuracy : {accuracy:.4f}"
)

# ==========================================
# Q9 : Complete Model Comparison
# ==========================================

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.markdown("---")
st.title("Q9 : Complete Model Comparison")

# -----------------------------
# Load Iris Dataset
# -----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

st.subheader("Dataset")

st.dataframe(X.head())

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Models
# -----------------------------

models = {

    "SVM": SVC(
        C=1,
        kernel="linear",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

}

# -----------------------------
# Train & Evaluate
# -----------------------------

results = []

best_model = ""
best_accuracy = 0

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    results.append({

        "Model": name,

        "Accuracy": round(
            accuracy,
            4
        )

    })

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = name

# -----------------------------
# Comparison Table
# -----------------------------

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
)

st.subheader("Model Comparison")

st.dataframe(comparison_df)

# -----------------------------
# Best Model
# -----------------------------

st.success(
    f"Best Model : {best_model}"
)

st.success(
    f"Accuracy : {best_accuracy:.4f}"
)

st.info(
    f"{best_model} achieved the highest accuracy among all models."
)

# ==========================================
# Q10 : Mini Project - Full Pipeline
# Part 1 : Import, Dataset & Train-Test Split
# ==========================================

import streamlit as st
import pandas as pd
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

st.set_page_config(
    page_title="Mini Project",
    layout="wide"
)

st.title("Question 10: Complete Machine Learning Pipeline")

# -----------------------------
# Load Dataset
# -----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

st.subheader("Dataset")

st.dataframe(X.head())

st.write("Dataset Shape :", X.shape)

st.write("Missing Values")

st.write(X.isnull().sum())

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

st.success("Train Test Split Completed")