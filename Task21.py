import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("ford_car_dataset.csv")

# Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# One-Hot Encoding
X = pd.get_dummies(X)

# Save columns
joblib.dump(X.columns.tolist(), "columns.pkl")

# Numerical columns
numerical_columns = [
    "year",
    "mileage",
    "tax",
    "mpg",
    "engineSize"
]

# Scale only numerical columns
scaler = StandardScaler()

X[numerical_columns] = scaler.fit_transform(
    X[numerical_columns]
)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "LR_ford_car.pkl")

print("Model Saved Successfully")