print("Question No. 2")

import pandas as pd

# तुझ्या CSV फाइलचे नाव इथे टाक
df = pd.read_csv("ford_car_dataset.csv")

# सर्व columns पाहण्यासाठी
print(df.columns)

# पहिले 5 records पाहण्यासाठी
print(df.head())

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Display the shape of the training and testing datasets
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)