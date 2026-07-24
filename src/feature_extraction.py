import cv2
import numpy as np


# =========================
# Chargement de l'image
# =========================

def load_image(image_path):
    image_path = str(image_path)

    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")

    return image


# =========================
# Pourcentage de rouille
# =========================

def extract_rust_percentage(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Intervalle des couleurs orangées/brunes
    lower_rust = np.array([5, 50, 50])
    upper_rust = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_rust, upper_rust)

    rust_pixels = np.count_nonzero(mask)
    total_pixels = mask.size

    pct_rust = rust_pixels / total_pixels

    return pct_rust


# =========================
# Rugosité (Sobel)
# =========================

def extract_roughness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    roughness = np.var(gradient)

    return roughness


# =========================
# Variable personnelle
# =========================

def extract_rust_blob_count(image):
    """
    Compte le nombre de taches de rouille présentes sur la feuille.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_rust = np.array([5, 50, 50])
    upper_rust = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_rust, upper_rust)

    # Nettoyage du masque pour supprimer le bruit
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Détection des contours des taches
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Ignorer les très petites taches (bruit)
    min_area = 20

    rust_blob_count = sum(
        1 for contour in contours
        if cv2.contourArea(contour) >= min_area
    )

    return rust_blob_count


# =========================
# Extraction complète
# =========================

def extract_features(image):

    pct_rust = extract_rust_percentage(image)
    roughness = extract_roughness(image)
    rust_blob_count = extract_rust_blob_count(image)

    return {
        "pct_rouille": pct_rust,
        "rugosite": roughness,
        "nb_taches_rouille": rust_blob_count
    }


# =========================
# Test
# =========================

if __name__ == "__main__":

    image = load_image("../dataset/saines/image1.jpg")

    features = extract_features(image)

    print(features)