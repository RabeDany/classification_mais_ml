import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================
# Chemins
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "Custom Decision Tree": MODELS_DIR / "custom_decision_tree.pkl",
    "Custom Random Forest": MODELS_DIR / "custom_random_forest.pkl",
    "Sklearn Decision Tree": MODELS_DIR / "sklearn_decision_tree.pkl",
    "Sklearn Random Forest": MODELS_DIR / "sklearn_random_forest.pkl"
}


# ==========================
# Chargement des données de test
# ==========================

def load_test_data():

    X_test = joblib.load(MODELS_DIR / "X_test.pkl")
    y_test = joblib.load(MODELS_DIR / "y_test.pkl")

    return X_test, y_test


# ==========================
# Evaluation d'un modèle
# ==========================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1-Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Confusion Matrix": confusion_matrix(
            y_test,
            predictions
        )
    }


# ==========================
# Evaluation de tous les modèles
# ==========================

def evaluate_models():

    X_test, y_test = load_test_data()

    results = []

    print("=" * 70)
    print("EVALUATION DES MODELES")
    print("=" * 70)

    for name, model_path in MODELS.items():

        model = joblib.load(model_path)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        print(f"\n{name}")
        print("-" * 70)
        print(f"Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"Precision : {metrics['Precision']:.4f}")
        print(f"Recall    : {metrics['Recall']:.4f}")
        print(f"F1 Score  : {metrics['F1-Score']:.4f}")

        print("\nConfusion Matrix")
        print(metrics["Confusion Matrix"])

        results.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"]
        })

        matrix_filename = (
            name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            + "_confusion_matrix.csv"
        )

        pd.DataFrame(
            metrics["Confusion Matrix"]
        ).to_csv(
            RESULTS_DIR / matrix_filename,
            index=False
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    results_df.to_csv(
        RESULTS_DIR / "evaluation_results.csv",
        index=False
    )

    print("\n")
    print("=" * 70)
    print("CLASSEMENT FINAL")
    print("=" * 70)

    print(results_df)

    print("\nRésultats enregistrés dans :")
    print(RESULTS_DIR)


if __name__ == "__main__":
    evaluate_models()