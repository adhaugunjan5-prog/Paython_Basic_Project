# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Students_Performance_Corrected (1).csv")

# -------------------------------
# Data Exploration (EDA)
# -------------------------------

print("========== First 5 Records ==========")
print(df.head())

print("\n========== Dataset Shape ==========")
print(df.shape)

print("\n========== Column Names ==========")
print(df.columns)

print("\n========== Data Types ==========")
print(df.dtypes)

print("\n========== Missing Values ==========")
print(df.isnull().sum())

print("\n========== Summary Statistics ==========")
print(df.describe())

# -------------------------------
# Histogram
# -------------------------------
plt.figure(figsize=(6,4))
plt.hist(df["Marks"], bins=10, edgecolor="black")
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# Bar Chart
# -------------------------------
plt.figure(figsize=(6,4))
df["Result"].value_counts().plot(kind="bar")
plt.title("Pass vs Fail Students")
plt.xlabel("Result")
plt.ylabel("Count")
plt.show()

# -------------------------------
# Scatter Plot
# -------------------------------
plt.figure(figsize=(6,4))
plt.scatter(df["StudyHours"], df["Marks"])
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# -------------------------------
# Box Plot
# -------------------------------
plt.figure(figsize=(6,4))
plt.boxplot(df["Marks"])
plt.title("Marks Box Plot")
plt.ylabel("Marks")
plt.show()

# -------------------------------
# Pie Chart
# -------------------------------
plt.figure(figsize=(6,6))
df["Gender"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

# -------------------------------
# Heatmap
# -------------------------------
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nEDA Completed Successfully!")