from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path

# Veri yolları
DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_FILE = DATA_DIR / "match_model.pkl"
ENCODER_FILE = DATA_DIR / "match_label_encoder.pkl"

# Model ve encoder'ı yükle
model = joblib.load(MODEL_FILE)
le = joblib.load(ENCODER_FILE)

# FastAPI uygulaması
app = FastAPI()

# Request modeli
class MatchRequest(BaseModel):
    label1: str
    label2: str

@app.post("/predict-score")
def predict_score(req: MatchRequest):
    try:
        enc1 = le.transform([req.label1])[0]
        enc2 = le.transform([req.label2])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz label ismi")

    feature = [[enc1, enc2, abs(enc1 - enc2)]]
    prediction = model.predict(feature)[0]

    return {"score": round(prediction, 2)}
