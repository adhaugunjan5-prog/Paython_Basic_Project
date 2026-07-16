import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Question No.3")

# Load Dataset
df = pd.read_csv("ford_cardataset (1).csv")

# Independent Features
X = df.drop("Price", axis=1)

# One-Hot Encoding for Categorical Columns
cat_columns = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=cat_columns)

# Numerical Columns
num_columns = ["year", "mileage", "tax", "mpg", "engineSize"]

# Standard Scaling
scaler = StandardScaler()
X[num_columns] = scaler.fit_transform(X[num_columns])

# Display first 5 rows
print(X.head())