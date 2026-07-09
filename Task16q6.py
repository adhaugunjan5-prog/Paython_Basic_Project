print("Question No.6")
import pandas as pd

# Read the dataset
df = pd.read_csv("insurance.csv")

import matplotlib.pyplot as plt
import seaborn as sns

categorical_columns = ['sex', 'smoker', 'region']

for column in categorical_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[column])
    plt.title(f"Count Plot of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()
