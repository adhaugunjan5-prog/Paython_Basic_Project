import pandas as pd
import matplotlib.pyplot as plt

print("Question No. 5")

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Select Numeric Columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Histograms
df[numeric_columns].hist(figsize=(15, 10), bins=20)

plt.tight_layout()
plt.show()