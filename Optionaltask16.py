import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("car_data.csv")

# Basic EDA
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.describe())
print(df.columns)
print(df.dtypes)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())

# Value Counts
print(df["Fuel_Type"].value_counts())
print(df["Brand"].value_counts())

# Histogram - Price
sns.histplot(df["Price"])
plt.title("Price Distribution")
plt.show()

# Histogram - Mileage
sns.histplot(df["Mileage"])
plt.title("Mileage Distribution")
plt.show()

# Count Plot
sns.countplot(x="Fuel_Type", data=df)
plt.show()

# Count Plot
sns.countplot(x="Transmission", data=df)
plt.show()

# Scatter Plot
sns.scatterplot(x="Mileage", y="Price", data=df)
plt.show()

# Bar Plot
sns.barplot(x="Fuel_Type", y="Price", data=df)
plt.show()

# Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()