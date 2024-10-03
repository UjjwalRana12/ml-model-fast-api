from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


diabetes_router=APIRouter(
    prefix="/diabetes",
    tags=['diabetes']
)

model = joblib.load("models\diabetes_model.pkl")
scaler=joblib.load("models\StandardscalerDiabetes.pkl")

class InputDiabetes(BaseModel):
    pregnancies:float
    glucose:float
    BloodPressure:float
    SkinThickness:float               
    Insulin :float                    
    BMI:float                         
    DiabetesPedigreeFunction:float    
    Age:float



@diabetes_router.post('/predict')
async def diabetes(data:InputDiabetes):
    input_data=[[data.pregnancies,
                 data.glucose,
                 data.BloodPressure,
                data.SkinThickness,
                data.Insulin,
                data.BMI,
                data.DiabetesPedigreeFunction,
                data.Age]]
    
    reshaped_input = np.array(input_data).reshape(1, -1)

    print("Reshaped Input:", reshaped_input)

    scaled_input = scaler.transform(reshaped_input)

    print("Scaled Input:", scaled_input)

    prediction = model.predict(scaled_input)

    print("Prediction:", prediction)

    return {"prediction": int(prediction[0])}
