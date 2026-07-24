import numpy as np

from src.metrics import (
    node_purity,
    split_dataset,
    find_best_feature
)


class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class MaxMinorityDecisionTree:

    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=0)

    def build_tree(self, X, y, depth):

        # Plus de données
        if len(y) == 0:
            return None

        # Nœud pur
        if node_purity(y) == 1:
            return Node(value=y[0])

        # Profondeur maximale atteinte
        if depth >= self.max_depth:
            return Node(value=self.majority_class(y))

        # Recherche du meilleur split
        feature, threshold, score = find_best_feature(X, y)

        # Aucun split trouvé
        if feature is None:
            return Node(value=self.majority_class(y))

        X_left, X_right, y_left, y_right = split_dataset(
            X,
            y,
            feature,
            threshold
        )

        # Sécurité
        if len(y_left) == 0 or len(y_right) == 0:
            return Node(value=self.majority_class(y))

        left_child = self.build_tree(
            X_left,
            y_left,
            depth + 1
        )

        right_child = self.build_tree(
            X_right,
            y_right,
            depth + 1
        )

        return Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def majority_class(self, y):

        classes, counts = np.unique(
            y,
            return_counts=True
        )

        return classes[np.argmax(counts)]

    def predict_sample(self, x, node):

        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)

        return self.predict_sample(x, node.right)

    def predict(self, X):

        return np.array([
            self.predict_sample(sample, self.root)
            for sample in X
        ])

    def score(self, X, y):
        """
        Calcule l'accuracy du modèle.
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy