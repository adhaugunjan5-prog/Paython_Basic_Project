import pandas as pd

# Load the Ford Car Dataset
df = pd.read_csv("ford_car_dataset.csv")

# Display original categorical columns (Before Encoding)
print("Before Encoding:")
print(df[["model", "transmission"]].head())

# Identify all categorical columns
categorical_columns = df.select_dtypes(include=["object"]).columns

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Display encoded columns (After Encoding)
print("\nAfter Encoding (First 5 Rows):")
print(df_encoded.head())

# Display dataset shape before and after encoding
print("\nShape Before Encoding:", df.shape)
print("Shape After Encoding :", df_encoded.shape)