import json
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Veri dizini
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "label_match_scores.json"

# Veriyi yükle
with open(DATA_FILE, "r", encoding="utf-8") as f:
    match_scores = json.load(f)

# Tüm örnekleri (label1, label2, score) olarak çıkar
data = []
for label1, inner in match_scores.items():
    for label2, score in inner.items():
        data.append((label1, label2, score))

# Label'ları sayısal verilere çevir (encoding)
labels = list(set([x[0] for x in data] + [x[1] for x in data]))
le = LabelEncoder()
le.fit(labels)

X = []
y = []

for label1, label2, score in data:
    enc1 = le.transform([label1]).tolist()[0] # type: ignore
    enc2 = le.transform([label2]).tolist()[0] # type: ignore
    encoded_pair = [enc1, enc2, abs(enc1 - enc2)]
    X.append(encoded_pair)
    y.append(score)

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test performansı
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Test MSE: {mse:.4f}")

# Modeli ve encoder'ı kaydet
joblib.dump(model, DATA_DIR / "match_model.pkl")
joblib.dump(le, DATA_DIR / "match_label_encoder.pkl")

print("✅ Eşleşme skoru modeli eğitildi ve kaydedildi.")
