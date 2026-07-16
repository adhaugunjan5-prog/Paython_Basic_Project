import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Question No.6")

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Select Categorical Columns
cat_columns = df.select_dtypes(include=["object"]).columns

# Count Plot for each Categorical Column
for col in cat_columns:
    plt.figure(figsize=(7,5))
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.title(col)
    plt.tight_layout()
    plt.show()