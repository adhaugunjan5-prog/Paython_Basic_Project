# Import pandas
import pandas as pd

# Load the Ford Car Dataset
# Replace 'ford.csv' with the correct file path if needed
df = pd.read_csv('ford_car_dataset.csv')

# Display the first 10 rows
print("First 10 Rows:")
print(df.head(10))

# Display the last 5 rows
print("\nLast 5 Rows:")
print(df.tail(5))

# Check the shape of the dataset
print("\nDataset Shape (Rows, Columns):")
print(df.shape)

# Check data types of all columns
print("\nData Types:")
print(df.dtypes)