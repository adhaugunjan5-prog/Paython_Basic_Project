import pandas as pd

# Read the dataset
df = pd.read_csv("insurance.csv")

print("Question No. 3")

print("\nMissing Values:")
print(df.isnull().sum())