import pandas as pd

# Read the dataset
df = pd.read_csv("insurance.csv")

print("Question No. 2")

print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())