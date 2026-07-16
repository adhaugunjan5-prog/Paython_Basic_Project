print("Question no.1")
import pandas as pd

# Load Dataset
df = pd.read_csv("ford_car_dataset (1).csv")

# Independent and Dependent Features
X = df.drop("price", axis=1)
Y = df["price"]

# Print Shapes
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
