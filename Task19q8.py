import pandas as pd

print("Question No. 8")

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Independent and Dependent Variables
X = df.drop("Price", axis=1)
y = df["Price"]

print("Independent Features:")
print(X.columns)

print("\nDependent Feature:")
print(y.name)