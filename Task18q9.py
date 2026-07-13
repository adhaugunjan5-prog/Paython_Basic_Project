from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the Ford Car Dataset
df = pd.read_csv("ford_car_dataset.csv")

# Apply One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# Separate Independent Features (X) and Dependent Feature (y)
X = df.drop("price", axis=1)
y = df["price"]

# Apply Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data into DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Display first 5 rows of scaled data
print("First 5 Rows of Scaled Data:")
print(X_scaled.head())