import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Ford Car Dataset
df = pd.read_csv("ford_car_dataset.csv")

# Select all categorical columns
categorical_columns = df.select_dtypes(include=["object"]).columns

# Create count plots for each categorical column
for column in categorical_columns:
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(f"Count Plot of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()