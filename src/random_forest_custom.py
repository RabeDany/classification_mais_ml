import numpy as np

from src.max_minority_tree import MaxMinorityDecisionTree


class MaxMinorityRandomForest:

    def __init__(
        self,
        n_estimators=100,
        max_depth=5
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def bootstrap_sample(self, X, y):
        """
        Génère un échantillon Bootstrap.
        """

        n_samples = len(X)

        indices = np.random.choice(
            n_samples,
            size=n_samples,
            replace=True
        )

        return X[indices], y[indices]

    def fit(self, X, y):

        self.trees = []

        for _ in range(self.n_estimators):

            X_sample, y_sample = self.bootstrap_sample(X, y)

            tree = MaxMinorityDecisionTree(
                max_depth=self.max_depth
            )

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

    def predict(self, X):

        if len(self.trees) == 0:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. Appelez fit() avant predict()."
            )

        # Chaque ligne correspond aux prédictions d'un arbre
        predictions = np.array([
            tree.predict(X)
            for tree in self.trees
        ])

        final_predictions = []

        # Vote majoritaire
        for i in range(X.shape[0]):

            votes = predictions[:, i]

            final_predictions.append(
                np.bincount(votes.astype(int)).argmax()
            )

        return np.array(final_predictions)

    def score(self, X, y):
        """
        Calcule l'accuracy du modèle.
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy