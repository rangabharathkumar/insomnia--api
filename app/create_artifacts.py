import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Create artifacts directory if it doesn't exist
artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

# Define numerical columns
NUMERICAL_COLS = ['Age', 'Sleep Duration', 'Quality of Sleep', 
                 'Physical Activity Level', 'Stress Level', 
                 'Heart Rate', 'Daily Steps']

# Sample data for creating scaler
data = {
    'Gender': ['Male', 'Female'],
    'Age': [25, 30],
    'Occupation': ['Software Engineer', 'Doctor'],
    'Sleep Duration': [7.5, 8.0],
    'Quality of Sleep': [7, 8],
    'Physical Activity Level': [5, 6],
    'Stress Level': [4, 3],
    'BMI Category': ['Normal', 'Overweight'],
    'Blood Pressure': ['120/80', '130/85'],
    'Heart Rate': [70, 75],
    'Daily Steps': [8000, 10000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create and save scaler with only numerical columns
scaler = StandardScaler()
scaler.fit(df[NUMERICAL_COLS])
joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))

print("Scaler created and saved successfully!") 