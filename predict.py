# predict.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Modelleri yükle
model = joblib.load("model/character_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

app = FastAPI()

class PredictionInput(BaseModel):
    answers: list  # 22 elemanlı, 1-3 arası sayılar

@app.post("/predict")
def predict_label(input_data: PredictionInput):
    if len(input_data.answers) != 22:
        return {"error": "22 adet cevap bekleniyor."}

    prediction = model.predict([input_data.answers])
    label = label_encoder.inverse_transform(prediction)[0]
    return {"label": label}
