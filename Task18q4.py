import pandas as pd
import matplotlib.pyplot as plt

# Load the Ford Car Dataset
df = pd.read_csv("ford_car_dataset.csv")

# Create histograms for important numeric columns
numeric_columns = ["price", "mileage", "year", "engineSize", "mpg"]

for column in numeric_columns:
    plt.figure(figsize=(6,4))
    plt.hist(df[column], bins=20, edgecolor='black')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()