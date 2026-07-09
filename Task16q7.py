print("Question No. 7")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv("insurance.csv")

plt.figure(figsize=(6,5))

sns.heatmap(
    df[['age', 'bmi', 'children', 'expenses']].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()