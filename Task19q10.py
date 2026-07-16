import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Question No.10")

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Encode Categorical Columns
encoded_df = pd.get_dummies(df, drop_first=True)

# Separate Features and Target
X = encoded_df.drop("Price", axis=1)
y = encoded_df["Price"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to DataFrame
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print(scaled_df.head())