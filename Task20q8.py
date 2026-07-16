import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

print("Question No.8")

# Load Dataset
df = pd.read_csv("ford_car_dataset (1).csv")

# Features and Target
X = df.drop("price", axis=1)      # जर column "Price" असेल तर Price लिहा
y = df["price"]

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "LR_ford_car.pkl")

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

# Save Feature Columns
joblib.dump(X.columns.tolist(), "columns.pkl")

print("Model and preprocessing files saved successfully.")