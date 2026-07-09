print("Question No.8")
import pandas as pd

# Read the dataset
df = pd.read_csv("insurance.csv")

print("Average Expenses:")
print(df['expenses'].mean())

print("\nMaximum Expenses:")
print(df['expenses'].max())

print("\nMinimum Expenses:")
print(df['expenses'].min())

print("\nAverage Expenses by Smoker:")
print(df.groupby('smoker')['expenses'].mean())
