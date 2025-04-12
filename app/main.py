# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model_utils import predict  # Ensure this path works when running from root
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Insomnia Prediction API")

class InputData(BaseModel):
    Gender: str
    Age: int
    Occupation: str
    Sleep_Duration: float
    Quality_of_Sleep: int
    Physical_Activity_Level: int
    Stress_Level: int
    BMI_Category: str
    Blood_Pressure: str
    Heart_Rate: int
    Daily_Steps: int

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Age": 30,
                "Occupation": "Software Engineer",
                "Sleep_Duration": 7.5,
                "Quality_of_Sleep": 8,
                "Physical_Activity_Level": 5,
                "Stress_Level": 4,
                "BMI_Category": "Normal",
                "Blood_Pressure": "120/80",
                "Heart_Rate": 70,
                "Daily_Steps": 8000
            }
        }

@app.post("/predict")
async def get_prediction(data: InputData):
    try:
        logger.info(f"Received prediction request with data: {data.dict()}")
        result = predict(data.dict())
        logger.info(f"Prediction result: {result}")
        return {"prediction": result}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
