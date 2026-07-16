import pandas as pd

print("Question No. 9")

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Before Encoding
print("Before Encoding")
categorical_df = df.select_dtypes(include=["object"])
print(categorical_df.head())

# One-Hot Encoding
encoded_df = pd.get_dummies(df, drop_first=True)

print("\nAfter Encoding")
print(encoded_df.head())