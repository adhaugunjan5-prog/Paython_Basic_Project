import pandas as pd

print("Question No.2")

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Independent Features
X = df.drop("Price", axis=1)

# Identify Categorical Columns
cat_columns = X.select_dtypes(include=["object"]).columns

print("Categorical Columns:")
print(cat_columns)

# One-Hot Encoding
X = pd.get_dummies(X, columns=cat_columns)

# Display first 5 rows
print("\nAfter Encoding:")
print(X.head())