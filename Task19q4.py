import pandas as pd

print("Question No.4")

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Summary Statistics
print(df.describe())

# Target Variable (Price)
print("\nMinimum Price:", df["Price"].min())
print("Maximum Price:", df["Price"].max())
print("Mean Price:", df["Price"].mean())
print("Median Price:", df["Price"].median())