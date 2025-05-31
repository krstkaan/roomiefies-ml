import json
from pathlib import Path

# Dosya yolunu belirt
DATA_DIR = Path(__file__).resolve().parent / "data"
MATCH_SCORE_FILE = DATA_DIR / "label_match_scores.json"

# Skorları belleğe al
with open(MATCH_SCORE_FILE, "r", encoding="utf-8") as f:
    match_scores = json.load(f)

def get_match_score(label1: str, label2: str) -> int:
    """
    İki karakter label'ı arasındaki eşleşme skorunu döner.
    Eğer tanımsızsa 0 döner.
    """
    return match_scores.get(label1, {}).get(label2, 0)

if __name__ == "__main__":
    print("partici vs sosyal_kelebek:", get_match_score("partici", "sosyal_kelebek"))
    print("muhafazakar vs yalniz_kurt:", get_match_score("muhafazakar", "yalniz_kurt"))
    print("minimalist vs sanatci_ruh:", get_match_score("minimalist", "sanatci_ruh"))
