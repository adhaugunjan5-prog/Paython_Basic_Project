import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Ford Car Dataset
df = pd.read_csv("ford_car_dataset.csv")

# Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5)

plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Correlation of all features with price
print("\nCorrelation with Price:")
print(corr_matrix["price"].sort_values(ascending=False))