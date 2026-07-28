import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("./Program_App/salary_prediction.csv")

# Select Features
features = [
    'Rating',
    'age',
    'python_yn',
    'R_yn',
    'spark',
    'aws',
    'excel',
    'desc_len',
    'num_comp'
]

X = df[features]
y = df['avg_salary']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Evaluation
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

# Save Files
joblib.dump(model, "LinearReg_salary.pkl")
joblib.dump(scaler, "scaler_salary.pkl")
joblib.dump(features, "columns.pkl")

print("Model Saved Successfully")