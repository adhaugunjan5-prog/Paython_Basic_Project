import streamlit as st
import pandas as pd
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score


st.title("Regression Algorithms Comparison")


# Load Dataset
@st.cache_data
def load_data():
    housing = fetch_california_housing()

    X = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    y = housing.target

    return X, y


X_reg, y_reg = load_data()


# Train-Test Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)


# Feature Scaling
scaler = StandardScaler()

X_train_reg = scaler.fit_transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)


# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor()
}

regression_models = models
regression_results = {}

results = []


# Training and Evaluation
for name, model in models.items():

    model.fit(X_train_reg, y_train_reg)

    y_pred = model.predict(X_test_reg)

    r2 = r2_score(y_test_reg, y_pred)

    regression_results[name] = r2

    results.append({
        "Model": name,
        "R² Score": round(r2, 4)
    })

    st.subheader(name)
    st.write("R² Score:", r2)


# Comparison Table
comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="R² Score",
    ascending=False
)


st.subheader("Model Comparison")
st.dataframe(comparison_df)


# Best Model
best_name = comparison_df.iloc[0]["Model"]

best_reg_model = models[best_name]


# Save Best Model
joblib.dump(
    best_reg_model,
    "best_regression_model.pkl"
)

joblib.dump(
    scaler,
    "regression_scaler.pkl"
)

joblib.dump(
    X_reg.columns.tolist(),
    "regression_columns.pkl"
)


st.success(
    f"Best Model: {best_name} saved successfully!"
)

#Question 4

# Q4: Best Model Selection & Saving

import joblib

# -----------------------------
# 1. Select Best Classification Model
# -----------------------------

# Example:
# classification_results = {
#     "Logistic Regression": accuracy_score,
#     "Random Forest": accuracy_score,
#     "SVM": accuracy_score
# }

best_classification_model_name = max(
    classification_results, 
    key=classification_results.get
)

best_classification_model = classification_models[best_classification_model_name]

print("Best Classification Model:", best_classification_model_name)
print("Score:", classification_results[best_classification_model_name])


# -----------------------------
# 2. Select Best Regression Model
# -----------------------------

# Example:
# regression_results = {
#     "Linear Regression": RMSE,
#     "Random Forest Regressor": RMSE,
#     "XGBoost": RMSE
# }

# For regression, lower RMSE/MAE is better
best_regression_model_name = max(
    regression_results,
    key=regression_results.get
)

best_regression_model = regression_models[best_regression_model_name]

print("Best Regression Model:", best_regression_model_name)
print("RMSE:", regression_results[best_regression_model_name])


# -----------------------------
# 3. Save Models using Joblib
# -----------------------------

# Save best classification model
joblib.dump(
    best_classification_model,
    "best_classification_model.pkl"
)

# Save best regression model
joblib.dump(
    best_regression_model,
    "best_regression_model.pkl"
)


# -----------------------------
# 4. Save Scaler (if used)
# -----------------------------

joblib.dump(
    scaler,
    "scaler.pkl"
)


# -----------------------------
# 5. Save Feature Columns
# -----------------------------

joblib.dump(
    X_train.columns.tolist(),
    "feature_columns.pkl"
)


print("Models, scaler, and feature columns saved successfully!")
# Load classification model
classification_model = joblib.load(
    "best_classification_model.pkl"
)

# Load regression model
regression_model = joblib.load(
    "best_regression_model.pkl"
)

# Load scaler
scaler = joblib.load(
    "scaler.pkl"
)

# Load columns
feature_columns = joblib.load(
    "feature_columns.pkl"
)

print("Models loaded successfully!")