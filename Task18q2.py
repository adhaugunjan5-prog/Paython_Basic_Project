import pandas as pd

# Load the Ford Car Dataset
df = pd.read_csv("ford_car_dataset.csv")

# Check missing values
print("Missing Values in Each Column:")
print(df.isnull().sum())

# Total missing values
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Check duplicate rows
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Verify duplicates are removed
print("\nDuplicate Rows After Removal:")
print(df.duplicated().sum())

# Dataset shape after removing duplicates
print("\nDataset Shape After Removing Duplicates:")
print(df.shape)