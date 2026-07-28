print("Question No. 4")

# Predict the target values for the test dataset
y_pred = model.predict(X_test)

# Display the first 10 actual values
print("Actual Values:")
print(y_test.head(10).values)

print()

# Display the first 10 predicted values
print("Predicted Values:")
print(y_pred[:10])