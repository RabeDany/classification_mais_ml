from pathlib import Path
import tempfile
import os

import streamlit as st

from src.predict import predict_image


# ==========================
# Chemins
# ==========================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_NAME_FILE = MODELS_DIR / "best_model_name.txt"


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

uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Image sélectionnée",
        width="stretch"
    )

    if st.button("Lancer la prédiction"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp:

            temp.write(uploaded_file.read())
            temp_path = temp.name

        try:

            result = predict_image(temp_path)

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