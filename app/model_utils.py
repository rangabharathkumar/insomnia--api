import os
import pandas as pd
import numpy as np
import joblib

# Define paths relative to this file's directory
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

# Load model and artifacts
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
scaler = joblib.load(SCALER_PATH)

def preprocess_input(data: dict) -> np.ndarray:
    """
    Preprocess incoming data: encode categorical features and scale numerical ones.
    """
    df = pd.DataFrame([data])

    # Apply label encoding to categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                # Handle unseen labels by setting a default or most common label
                df[col] = le.transform([le.classes_[0]])

    # Apply scaling
    df_scaled = scaler.transform(df)

    return df_scaled

def predict(data: dict) -> str:
    """
    Preprocess data and return model prediction.
    """
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)[0]

    # Optional: convert prediction to label
    if hasattr(model, "classes_"):
        prediction = model.classes_[prediction] if isinstance(prediction, (int, np.integer)) else prediction

    return str(prediction)
