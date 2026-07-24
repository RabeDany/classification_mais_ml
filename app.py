from pathlib import Path
import tempfile
import os
import json
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from src.predict import predict_image


# ==========================
# Chemins
# ==========================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_NAME_FILE = MODELS_DIR / "best_model_name.txt"
UPLOADS_DIR = BASE_DIR / "uploads"
HISTORY_FILE = UPLOADS_DIR / "history.json"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    return []


def save_history(history):
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def append_history(entry):
    history = load_history()
    history.insert(0, entry)
    save_history(history[:100])
    return history


def build_thumbnail(image_path, size=(250, 250)):
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        image = None

    if image is None:
        return None

    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (size[1] - new_h) // 2
    bottom = size[1] - new_h - top
    left = (size[0] - new_w) // 2
    right = size[0] - new_w - left

    thumbnail = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
    return thumbnail


# ==========================
# Configuration Streamlit
# ==========================

st.set_page_config(
    page_title="Classification des feuilles de maïs",
    page_icon="🌽",
    layout="centered"
)


# ==========================
# Chargement du meilleur modèle
# ==========================

if MODEL_NAME_FILE.exists():

    with open(MODEL_NAME_FILE, "r", encoding="utf-8") as file:
        MODEL_NAME = file.read().strip()

else:

    MODEL_NAME = "Inconnu"


# ==========================
# Interface
# ==========================

st.title("🌽 Détection de la rouille du maïs")

st.write(
    """
Cette application détecte automatiquement si une feuille de maïs est saine
ou malade grâce à un modèle de Machine Learning.
"""
)

st.divider()


# ==========================
# Upload de l'image
# ==========================

def render_history_section(history):
    st.subheader("Galerie d'historique des détections")

    if not history:
        st.info("Aucune détection historique n'est encore disponible.")
        return

    rows = [history[i:i + 3] for i in range(0, min(len(history), 9), 3)]

    for row in rows:
        cols = st.columns(3)

        for col, entry in zip(cols, row):
            with col:
                image_path = UPLOADS_DIR / entry["filename"]

                if image_path.exists():
                    thumbnail = build_thumbnail(image_path, size=(250, 250))
                    if thumbnail is not None:
                        st.image(
                            thumbnail,
                            caption=f"{entry['created_at']} - {entry['label']}",
                            width=250
                        )
                    else:
                        st.warning(f"Impossible de générer la vignette : {entry['filename']}")
                else:
                    st.warning(f"Image manquante : {entry['filename']}")

                st.markdown(
                    f"**Rouille :** {entry['pct_rouille']:.2%}  \n"
                    f"**Rugosité :** {entry['rugosite']:.2f}  \n"
                    f"**Taches :** {entry['nb_taches_rouille']}"
                )

                if entry["prediction"] == 1:
                    st.error("Malade")
                else:
                    st.success("Saine")

uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png"]
)

history = load_history()

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Image sélectionnée",
        width="stretch"
    )

    if st.button("Lancer la prédiction"):

        file_bytes = uploaded_file.getvalue()
        suffix = Path(uploaded_file.name).suffix or ".jpg"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_name = f"{Path(uploaded_file.name).stem}_{timestamp}{suffix}"
        saved_path = UPLOADS_DIR / saved_name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temp.write(file_bytes)
            temp_path = temp.name

        try:

            result = predict_image(temp_path)

            with open(saved_path, "wb") as saved_file:
                saved_file.write(file_bytes)

            history_entry = {
                "filename": saved_name,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": int(result["prediction"]),
                "label": result["label"],
                "pct_rouille": float(result["features"]["pct_rouille"]),
                "rugosite": float(result["features"]["rugosite"]),
                "nb_taches_rouille": int(result["features"]["nb_taches_rouille"])
            }

            history = append_history(history_entry)

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)

        st.divider()

        st.subheader("Résultat")

        if result["prediction"] == 1:

            st.error("🌿 Feuille malade")

        else:

            st.success("🌿 Feuille saine")

        st.divider()

        st.subheader("Caractéristiques extraites")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Pourcentage de rouille",
                f"{result['features']['pct_rouille']:.2%}"
            )

        with col2:

            st.metric(
                "Rugosité",
                f"{result['features']['rugosite']:.2f}"
            )

        with col3:

            st.metric(
                "Nombre de taches",
                result["features"]["nb_taches_rouille"]
            )

        st.divider()

        st.info(
            f"Modèle utilisé : **{MODEL_NAME}**"
        )

render_history_section(history)