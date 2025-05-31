from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

# ---- Karakter tahmini için yükleme ----
character_model = joblib.load("model/character_model.pkl")
character_label_encoder = joblib.load("model/label_encoder.pkl")

# ---- Eşleşme skoru modeli için yükleme ----
DATA_DIR = Path(__file__).resolve().parent / "data"
match_model = joblib.load(DATA_DIR / "match_model.pkl")
match_label_encoder = joblib.load(DATA_DIR / "match_label_encoder.pkl")


# ---- Giriş modelleri ----
class PredictionInput(BaseModel):
    answers: list  # 22 elemanlı, 1-3 arası sayılar

class MatchScoreInput(BaseModel):
    label1: str
    label2: str


# ---- Karakter tahmini endpoint ----
@app.post("/predict")
def predict_label(input_data: PredictionInput):
    if len(input_data.answers) != 22:
        raise HTTPException(status_code=400, detail="22 adet cevap bekleniyor.")
    
    prediction = character_model.predict([input_data.answers])
    label = character_label_encoder.inverse_transform(prediction)[0]
    return {"label": label}


# ---- Eşleşme skoru tahmini endpoint ----
@app.post("/predict-score")
def predict_score(data: MatchScoreInput):
    try:
        enc1 = match_label_encoder.transform([data.label1])[0]
        enc2 = match_label_encoder.transform([data.label2])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz label ismi")
    
    feature = [[enc1, enc2, abs(enc1 - enc2)]]
    prediction = match_model.predict(feature)[0]
    return {"score": round(prediction, 2)}
