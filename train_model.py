import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("⏳ Veri seti yükleniyor...")

# Yeni CSV'yi oku
df = pd.read_csv("data/train_dataset_10labels.csv")

print("✅ CSV başarıyla yüklendi.")

# Özellikleri ve etiketleri ayır
X = df.drop("label", axis=1)
y = df["label"]

# Etiketleri encode et
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("🧠 Model eğitiliyor...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

print("💾 Model ve label encoder kaydediliyor...")
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/character_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("✅ Eğitim tamamlandı ve model kaydedildi.")
