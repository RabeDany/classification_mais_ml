import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.max_minority_tree import MaxMinorityDecisionTree
from src.random_forest_custom import MaxMinorityRandomForest


# ==========================
# Chemins
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "features_dataset.csv"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# ==========================
# Configuration
# ==========================

FEATURE_COLUMNS = [
    "pct_rouille",
    "rugosite",
    "nb_taches_rouille"
]

TARGET_COLUMN = "label_malade"

MODEL_CONFIG = {
    "custom_decision_tree": MaxMinorityDecisionTree(max_depth=5),

    "custom_random_forest": MaxMinorityRandomForest(
        n_estimators=20,
        max_depth=5
    ),

    "sklearn_decision_tree": DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    ),

    "sklearn_random_forest": RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        random_state=42
    )
}


# ==========================
# Chargement des données
# ==========================

def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# ==========================
# Sauvegarde
# ==========================

def save_model(model, filename):

    joblib.dump(
        model,
        MODELS_DIR / filename
    )


# ==========================
# Entraînement
# ==========================

def train_models():

    X_train, X_test, y_train, y_test = load_dataset()

    scores = {}

    best_model = None
    best_model_name = None
    best_score = -1

    print("=" * 60)
    print("ENTRAINEMENT DES MODELES")
    print("=" * 60)

    for model_name, model in MODEL_CONFIG.items():

        print(f"\nEntraînement : {model_name}")

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        scores[model_name] = accuracy

        print(f"Accuracy : {accuracy:.4f}")

        save_model(
            model,
            f"{model_name}.pkl"
        )

        if accuracy > best_score:

            best_score = accuracy
            best_model = model
            best_model_name = model_name

    # Sauvegarde du meilleur modèle
    save_model(
        best_model,
        "best_model.pkl"
    )

    # Nom du meilleur modèle
    with open(
        MODELS_DIR / "best_model_name.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(best_model_name)

    # Jeu de test
    save_model(
        X_test,
        "X_test.pkl"
    )

    save_model(
        y_test,
        "y_test.pkl"
    )

    # Résultats
    results = pd.DataFrame({
        "Model": scores.keys(),
        "Accuracy": scores.values()
    })

    results = results.sort_values(
        by="Accuracy",
        ascending=False
    )

    results.to_csv(
        MODELS_DIR / "training_results.csv",
        index=False
    )

    print("\n")
    print("=" * 60)
    print("CLASSEMENT DES MODELES")
    print("=" * 60)

    print(results)

    print("\nMeilleur modèle :", best_model_name)
    print("Accuracy :", round(best_score, 4))


if __name__ == "__main__":
    train_models()