print("Question No. 9")
import pandas as pd

# Read the dataset
df = pd.read_csv("insurance.csv")

import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot with Smoker
plt.figure(figsize=(6,4))
sns.boxplot(x='smoker', y='expenses', data=df)
plt.title("Expenses by Smoker")
plt.xlabel("Smoker")
plt.ylabel("Expenses")
plt.show()

# Boxplot with Sex
plt.figure(figsize=(6,4))
sns.boxplot(x='sex', y='expenses', data=df)
plt.title("Expenses by Sex")
plt.xlabel("Sex")
plt.ylabel("Expenses")
plt.show()
