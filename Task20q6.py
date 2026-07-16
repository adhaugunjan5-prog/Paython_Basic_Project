import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Question No.6")

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("First 10 Predicted Values:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)
