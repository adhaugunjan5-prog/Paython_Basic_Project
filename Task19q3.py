import pandas as pd

print("Question No. 3")

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Missing Percentage
print("\nMissing Percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

# Fill Numeric Missing Values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill Categorical Missing Values
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove Duplicates
df = df.drop_duplicates()
print("New Shape:", df.shape)