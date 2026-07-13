import pandas as pd

# Load the Ford Car Dataset
df = pd.read_csv("ford_car_dataset.csv")

# Statistical summary of all numeric columns
print("Statistical Summary:")
print(df.describe())

# Price statistics
print("\nPrice Statistics")
print("Minimum Price :", df["price"].min())
print("Maximum Price :", df["price"].max())
print("Mean Price    :", df["price"].mean())
print("Median Price  :", df["price"].median())

# Mileage statistics
print("\nMileage Statistics")
print("Minimum Mileage :", df["mileage"].min())
print("Maximum Mileage :", df["mileage"].max())
print("Mean Mileage    :", df["mileage"].mean())
print("Median Mileage  :", df["mileage"].median())

# Year statistics
print("\nYear Statistics")
print("Minimum Year :", df["year"].min())
print("Maximum Year :", df["year"].max())
print("Mean Year    :", df["year"].mean())
print("Median Year  :", df["year"].median())