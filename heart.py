from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import joblib

heart_router = APIRouter(
    prefix="/heart",
    tags=['heart']
)


model = joblib.load("models/heart_attack_model.pkl")
scaler = joblib.load("models/scaler_heart.pkl")

class InputDataHeart(BaseModel):
    age: float
    sex: float
    cp: float
    trtbps: float
    chol: float
    fbs: float
    restecg: float
    thalachh: float
    exng: float
    oldpeak: float
    slp: float
    caa: float
    thall: float

@heart_router.post('/predict')
async def predict_heart(data: InputDataHeart):
    try:
        
        input_data = np.array([[data.age, data.sex, data.cp, data.trtbps, data.chol,
                                 data.fbs, data.restecg, data.thalachh, data.exng,
                                 data.oldpeak, data.slp, data.caa, data.thall]])

        print("Input Data:", input_data)  

        
        scaled_input = scaler.transform(input_data)

        print("Scaled Input:", scaled_input) 

        
        prediction = model.predict(scaled_input)

        print("Prediction:", prediction)  

        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
