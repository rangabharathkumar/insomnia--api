import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Create artifacts directory if it doesn't exist
artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

# Sample data for creating scaler
data = {
    'Gender': ['Male', 'Female'],
    'Age': [25, 30],
    'Occupation': ['Software Engineer', 'Doctor'],
    'Sleep_Duration': [7.5, 8.0],
    'Quality_of_Sleep': [7, 8],
    'Physical_Activity_Level': [5, 6],
    'Stress_Level': [4, 3],
    'BMI_Category': ['Normal', 'Overweight'],
    'Blood_Pressure': ['120/80', '130/85'],
    'Heart_Rate': [70, 75],
    'Daily_Steps': [8000, 10000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create and save scaler
scaler = StandardScaler()
scaler.fit(df.select_dtypes(include=[np.number]))
joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))

print("Scaler created and saved successfully!") 