import joblib
from pathlib import Path

# Veri dizini ve model dosyaları
DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_FILE = DATA_DIR / "match_model.pkl"
ENCODER_FILE = DATA_DIR / "match_label_encoder.pkl"

# Model ve encoder'ı yükle
model = joblib.load(MODEL_FILE)
le = joblib.load(ENCODER_FILE)

def predict_match_score(label1: str, label2: str) -> float:
    """
    İki karakter label'ı için eşleşme skorunu tahmin eder.
    """
    try:
        enc1 = le.transform([label1])[0]
        enc2 = le.transform([label2])[0]
    except ValueError as e:
        print(f"❌ Bilinmeyen label: {e}")
        return 0.0

    features = [[enc1, enc2, abs(enc1 - enc2)]]
    score = model.predict(features)[0]
    return round(score, 2)

if __name__ == "__main__":
    print("iskolik vs sosyal_kelebek:", predict_match_score("iskolik", "sosyal_kelebek"))
    print("minimalist vs evcimen:", predict_match_score("minimalist", "evcimen"))
    print("partici vs yalniz_kurt:", predict_match_score("partici", "yalniz_kurt"))
