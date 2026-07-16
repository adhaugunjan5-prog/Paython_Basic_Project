import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("Question No.7")

# Load Dataset
df = pd.read_csv("ford_car_dataset (1).csv")

# Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# R2 Score
r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)