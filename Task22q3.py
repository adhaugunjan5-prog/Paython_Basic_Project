print("Question No. 3")

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression Model
model = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    random_state=42
)

# Train the Model
model.fit(X_train, y_train)

print("Model Trained Successfully")