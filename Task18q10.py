# ===============================
# Ford Car Dataset - Complete Preprocessing Pipeline
# ===============================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load the Dataset
# -------------------------------
df = pd.read_csv("ford_car_dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# -------------------------------
# Handle Missing Values
# -------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values (if any)
df = df.dropna()

# -------------------------------
# Handle Duplicate Values
# -------------------------------
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

print("\nShape After Cleaning:")
print(df.shape)

# ===============================
# 2. Exploratory Data Analysis (EDA)
# ===============================

# ---------- Histograms ----------
numeric_cols = ["price", "mileage", "year", "engineSize", "mpg"]

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=20, edgecolor="black")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# ---------- Count Plots ----------
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(data=df,
                  x=col,
                  order=df[col].value_counts().index)

    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------- Correlation Heatmap ----------
plt.figure(figsize=(8,6))

corr = df.select_dtypes(include=["int64","float64"]).corr()

sns.heatmap(corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f")

plt.title("Correlation Heatmap")
plt.show()

print("\nCorrelation with Price:")
print(corr["price"].sort_values(ascending=False))

# ===============================
# 3. Identify Input & Output Features
# ===============================

X = df.drop("price", axis=1)
y = df["price"]

print("\nIndependent Features:")
print(X.columns)

print("\nDependent Feature:")
print(y.name)

# ===============================
# 4. One Hot Encoding
# ===============================

print("\nBefore Encoding:")
print(df[["model","transmission"]].head())

X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())

# ===============================
# 5. Feature Scaling
# ===============================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled,
                        columns=X.columns)

print("\nFirst 5 Rows of Scaled Data:")
print(X_scaled.head())

print("\nPreprocessing Completed Successfully!")