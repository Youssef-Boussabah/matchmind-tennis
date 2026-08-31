"""Random forest classifier built from :class:`DecisionTree`."""

from __future__ import annotations

import numpy as np

from .decision_tree import (
    DecisionTree,
    _check_int,
    _check_max_features,
    _check_non_negative_float,
    _check_random_state,
    _check_X,
    _check_Xy,
)


class RandomForest:
    """A bagged ensemble of decision trees for binary targets.

    Each tree is fitted on a bootstrap sample (unless ``bootstrap=False``) and
    considers a random subset of features at every split. Predictions come from
    averaging the trees' class probabilities.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: int | str | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ):
        if not isinstance(bootstrap, bool):
            raise ValueError("bootstrap must be True or False")
        if max_depth is not None:
            max_depth = _check_int(max_depth, "max_depth", minimum=0)

        self.n_estimators = _check_int(n_estimators, "n_estimators", minimum=1)
        self.max_depth = max_depth
        self.min_samples_split = _check_int(min_samples_split, "min_samples_split", minimum=2)
        self.min_impurity_decrease = _check_non_negative_float(
            min_impurity_decrease, "min_impurity_decrease"
        )
        self.max_features = _check_max_features(max_features)
        self.bootstrap = bootstrap
        self.random_state = _check_random_state(random_state)

        self.estimators: list[DecisionTree] = []
        self.n_features_: int | None = None

    def fit(self, X, y) -> RandomForest:
        X, y = _check_Xy(X, y)
        self.n_features_ = X.shape[1]
        self.estimators = []

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        # One seed per tree, drawn up front so each tree's feature sampling is
        # reproducible and independent of the order trees are fitted in.
        seeds = rng.integers(0, 2**32 - 1, size=self.n_estimators)

        for seed in seeds:
            if self.bootstrap:
                rows = rng.integers(0, n_samples, size=n_samples)
                X_sample, y_sample = X[rows], y[rows]
            else:
                X_sample, y_sample = X, y

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                random_state=int(seed),
            )
            self.estimators.append(tree.fit(X_sample, y_sample))

        return self

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        if not self.estimators:
            raise RuntimeError("this RandomForest is not fitted yet; call fit() first")
        X = _check_X(X, n_features=self.n_features_, allow_1d=True)
        probabilities = np.stack([tree.predict_proba(X) for tree in self.estimators])
        return probabilities.mean(axis=0)

    def score(self, X, y) -> float:
        X, y = _check_Xy(X, y)
        return float(np.mean(self.predict(X) == y))
