import numpy as np


def node_purity(y):
    """
    Calcule la pureté Max-Minority d'un nœud.
    """

    if len(y) == 0:
        return 0

    classes, counts = np.unique(y, return_counts=True)

    return np.max(counts) / len(y)


def weighted_purity(left_labels, right_labels):
    """
    Pureté pondérée après un split.
    """

    total = len(left_labels) + len(right_labels)

    return (
        len(left_labels) / total * node_purity(left_labels)
        + len(right_labels) / total * node_purity(right_labels)
    )


def split_dataset(X, y, feature_index, threshold):
    """
    Sépare le dataset selon un seuil.
    """

    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold

    return (
        X[left_mask],
        X[right_mask],
        y[left_mask],
        y[right_mask]
    )


def find_best_split(X_column, y):
    """
    Recherche le meilleur seuil d'une variable.
    """

    unique_values = np.unique(X_column)

    if len(unique_values) < 2:
        return None, -1

    thresholds = (
        unique_values[:-1] + unique_values[1:]
    ) / 2

    best_threshold = None
    best_score = -1

    for threshold in thresholds:

        left = y[X_column <= threshold]
        right = y[X_column > threshold]

        if len(left) == 0 or len(right) == 0:
            continue

        score = weighted_purity(left, right)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def find_best_feature(X, y):
    """
    Recherche la meilleure variable ainsi que son seuil.
    """

    best_feature = None
    best_threshold = None
    best_score = -1

    n_features = X.shape[1]

    for feature in range(n_features):

        threshold, score = find_best_split(
            X[:, feature],
            y
        )

        if score > best_score:
            best_score = score
            best_feature = feature
            best_threshold = threshold

    return best_feature, best_threshold, best_score