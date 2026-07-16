import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Question No. 7")

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()