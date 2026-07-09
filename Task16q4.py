print("Question No. 4")
import pandas as pd

# Read the dataset
df = pd.read_csv("insurance.csv")

numerical_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(include=['object', 'string']).columns

print("Numerical Columns:")
print(list(numerical_columns))

print("\nCategorical Columns:")
print(list(categorical_columns))

