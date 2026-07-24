import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pathlib import Path

import pandas as pd

from src.feature_extraction import load_image, extract_features


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_FILE = DATASET_DIR / "features_dataset.csv"


def process_folder(folder_path: Path, label: int):

    rows = []

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Le dossier '{folder_path}' est introuvable."
        )

    image_files = [
        file
        for file in folder_path.iterdir()
        if file.suffix.lower() in VALID_EXTENSIONS
    ]

    print(f"{folder_path.name} : {len(image_files)} image(s) trouvée(s).")

    for image_path in image_files:

        try:

            image = load_image(str(image_path))

            features = extract_features(image)

            rows.append({
                "id_image": image_path.name,
                "pct_rouille": features["pct_rouille"],
                "rugosite": features["rugosite"],
                "nb_taches_rouille": features["nb_taches_rouille"],
                "label_malade": label
            })

        except Exception as e:

            print(f"Erreur avec {image_path.name} : {e}")

    return rows


def build_dataset():

    healthy_folder = DATASET_DIR / "saines"
    sick_folder = DATASET_DIR / "malades"

    data = []

    data.extend(process_folder(healthy_folder, 0))
    data.extend(process_folder(sick_folder, 1))

    if len(data) == 0:
        raise ValueError(
            "Aucune image valide n'a été trouvée dans le dataset."
        )

    df = pd.DataFrame(data)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return df


if __name__ == "__main__":

    print("=" * 60)
    print("CREATION DU DATASET")
    print("=" * 60)

    df = build_dataset()

    print("\nDataset créé avec succès.")

    print(f"Nombre d'images : {len(df)}")

    print(f"Fichier enregistré : {OUTPUT_FILE}")