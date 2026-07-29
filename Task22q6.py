import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# Load dataset
df = pd.read_csv("your_dataset.csv")


# Features and Target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LogisticRegression()


# Train
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Evaluation
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))


print("\nClassification Report\n")
print(classification_report(y_test, y_pred))