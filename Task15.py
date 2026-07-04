import numpy as np
import pandas as pd

#from google.colab import files

# Upload CSV file
#uploaded = files.upload()

# Read the CSV file
path = "Countries.csv"
df = pd.read_csv(path)

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display last 5 rows
print("\nLast 5 Rows:")
print(df.tail())

# Shape
print("\nShape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Dataset information
print("\nInfo:")
df.info()

# Statistical summary
print("\nSummary:")
print(df.describe())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Unique values
print("\nUnique Values:")
print(df.nunique())

# Random 5 rows
print("\nRandom 5 Rows:")
print(df.sample(5))

# Value counts (replace 'Country' with your column name if needed)
print("\nValue Counts:")
print(df["country"].value_counts())

