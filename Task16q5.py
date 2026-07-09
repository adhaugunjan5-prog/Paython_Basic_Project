import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Question No. 5")

# Read the dataset
df = pd.read_csv("insurance.csv")

# Get all numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Plot histogram for each numerical column
for column in numerical_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()