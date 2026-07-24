import sys
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_extraction import (
    load_image,
    extract_features
)

import numpy as np

# ==========================
# Chemins
# ==========================

MODELS_DIR = ROOT_DIR / "models"
UPLOADS_DIR = ROOT_DIR / "uploads"

MODEL_PATH = MODELS_DIR / "best_model.pkl"


# ==========================
# Chargement du modèle
# ==========================

def load_model():
    return joblib.load(MODEL_PATH)


# ==========================
# Prédiction
# ==========================

def predict_image(image_path):

    image = load_image(str(image_path))

    features = extract_features(image)

    X = np.array([[
        features["pct_rouille"],
        features["rugosite"],
        features["nb_taches_rouille"]
    ]])

    model = load_model()

    prediction = model.predict(X)[0]

    return {
        "prediction": prediction,
        "label": "Malade" if prediction == 1 else "Saine",
        "features": features
    }


# ==========================
# Test
# ==========================

if __name__ == "__main__":

    image_path = UPLOADS_DIR / "test.jpg"

    result = predict_image(image_path)

    print("=" * 50)
    print("RESULTAT DE LA PREDICTION")
    print("=" * 50)

    print(f"Classe prédite : {result['label']}\n")

    print("Caractéristiques extraites :")

    for key, value in result["features"].items():
        print(f"{key} : {value}")