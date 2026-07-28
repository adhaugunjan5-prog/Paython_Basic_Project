print("Question No. 1")

import pandas as pd

# Load Dataset
df = pd.read_csv("heart.csv")

# Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Display Shape
print("X Shape:", X.shape)
print("y Shape:", y.shape)