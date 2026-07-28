print("Question No. 7")

import joblib

# Save the trained model
joblib.dump(model, "heart_model.pkl")

# Save the feature column names
joblib.dump(X.columns.tolist(), "columns.pkl")

print("Files Saved Successfully")