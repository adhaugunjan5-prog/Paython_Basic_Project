print("Question No. 8")

import pandas as pd
import joblib

# Load the saved model and column names
model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

# Create sample input data
sample = {
    "Age": 52,
    "RestingBP": 130,
    "Cholesterol": 250,
    "FastingBS": 0,
    "MaxHR": 150,
    "Oldpeak": 1.2,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingECG": "Normal",
    "ExerciseAngina": "N",
    "ST_Slope": "Up"
}

# Convert the sample data into a DataFrame
sample_df = pd.DataFrame([sample])

# Apply One-Hot Encoding
sample_df = pd.get_dummies(sample_df)

# Match the columns with the training data
sample_df = sample_df.reindex(
    columns=columns,
    fill_value=0
)

# Predict Heart Disease
prediction = model.predict(sample_df)

# Display the prediction result
if prediction[0] == 1:
    print("Heart Disease: YES")
else:
    print("Heart Disease: NO")