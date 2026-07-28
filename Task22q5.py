print("Question No. 5")

from sklearn.metrics import confusion_matrix

# Create the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Display the Confusion Matrix
print(cm)

# Extract TN, FP, FN, TP values
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

# Display the values
print("\nTN =", TN)
print("FP =", FP)
print("FN =", FN)
print("TP =", TP)