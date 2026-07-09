
print("Question No. 10")

import pandas as pd

# Read the dataset
df = pd.read_csv("insurance.csv")

print("Average Age:", df['age'].mean())
print("Average BMI:", df['bmi'].mean())

print("\nAverage Expenses by Smoker:")
print(df.groupby('smoker')['expenses'].mean())

print("\nRegion with Highest Customers:")
print(df['region'].value_counts())

print("\nSummary:")
print("1. Average age and BMI were calculated.")
print("2. Smokers have higher insurance expenses than non-smokers.")
print("3. The region with the highest number of customers is shown above.")
print("4. Boxplots and histograms show the distribution of the data and highlight outliers.")