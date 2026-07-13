import pandas as pd

# Load the dataset
df = pd.read_csv("ford_car_dataset.csv")

# Dependent (Target) Feature
target = "price"

# Independent (Input) Features
features = [col for col in df.columns if col != target]

print("Independent Features (Input):")
print(features)

print("\nDependent Feature (Output):")
print(target)