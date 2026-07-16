import pandas as pd
from sklearn.model_selection import train_test_split

print("Question No.4")

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

print("X_train Shape:", X_train.shape)
print("X_test Shape :", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape :", y_test.shape)