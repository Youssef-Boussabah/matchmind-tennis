"""Binary classification decision tree (CART, Gini impurity)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class _Node:
    """A leaf when `feature` is None, otherwise an internal split node."""

    prediction: int
    probability: float
    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def _gini(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    p1 = float(np.mean(y))
    return 1.0 - p1**2 - (1.0 - p1) ** 2


def _check_int(value, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _check_non_negative_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _check_max_features(value):
    if value is None or value == "sqrt":
        return value
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("max_features must be None, 'sqrt', or a positive integer")
    return int(value)


def _check_random_state(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"random_state must be None or a non-negative integer, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError("random_state must be non-negative")
    return int(value)


def _check_X(X, *, n_features: int | None = None, allow_1d: bool = False) -> np.ndarray:
    """Validate a feature matrix. With `allow_1d`, a single sample may be given as 1-D."""
    X = np.asarray(X)
    if np.iscomplexobj(X):
        raise ValueError("X must contain real-valued features")
    X = X.astype(float)
    if allow_1d and X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional (n_samples, n_features), got {X.ndim}")
    if X.size == 0:
        raise ValueError("X is empty")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"expected {n_features} features, got {X.shape[1]}")
    return X


def _check_Xy(X, y) -> tuple[np.ndarray, np.ndarray]:
    X = _check_X(X)
    y = np.asarray(y)
    if np.iscomplexobj(y):
        raise ValueError("y must contain real-valued labels")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}")
    # Check the values as given: casting first would silently round 0.2 to 0.
    values = y.astype(float)
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("y must contain only the binary labels 0 and 1")
    return X, values.astype(int)


def _best_threshold(
    column: np.ndarray, y: np.ndarray, parent_impurity: float
) -> tuple[float, float] | None:
    """Lowest threshold with the largest impurity decrease for one feature, and that decrease.

    Sorting the column once turns every candidate split into a prefix of the sorted
    samples, so the class counts on each side can be read straight off a cumulative
    sum instead of masking and re-counting the whole column for every threshold.
    """
    order = np.argsort(column, kind="stable")
    values = column[order]
    positives = np.cumsum(y[order])
    n = values.size

    # Candidates are the midpoints between adjacent distinct values.
    boundary = np.flatnonzero(values[:-1] < values[1:])
    if boundary.size == 0:
        return None
    thresholds = (values[boundary] + values[boundary + 1]) / 2.0

    # A midpoint can round onto the larger of the two values it came from, so count
    # the left side rather than assuming it ends at the boundary.
    n_left = np.searchsorted(values, thresholds, side="right")
    usable = (n_left > 0) & (n_left < n)
    if not usable.any():
        return None
    thresholds, n_left = thresholds[usable], n_left[usable]

    n_right = n - n_left
    left_positive = positives[n_left - 1]
    right_positive = positives[-1] - left_positive
    p_left = left_positive / n_left
    p_right = right_positive / n_right
    weighted = (
        n_left * (1.0 - p_left**2 - (1.0 - p_left) ** 2)
        + n_right * (1.0 - p_right**2 - (1.0 - p_right) ** 2)
    ) / n
    decrease = parent_impurity - weighted

    # argmax takes the first maximum, so equally good thresholds resolve to the lowest.
    first_best = int(np.argmax(decrease))
    return float(thresholds[first_best]), float(decrease[first_best])


def _resolve_max_features(max_features, n_features: int) -> int:
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(math.sqrt(n_features)))
    return min(int(max_features), n_features)


class DecisionTree:
    """A decision tree classifier for binary targets.

    Splits are chosen by exhaustive search over midpoints between adjacent unique
    feature values, using ``feature <= threshold`` for the left child throughout.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: int | str | None = None,
        random_state: int | None = None,
    ):
        if max_depth is not None:
            max_depth = _check_int(max_depth, "max_depth", minimum=0)

        self.max_depth = max_depth
        self.min_samples_split = _check_int(min_samples_split, "min_samples_split", minimum=2)
        self.min_impurity_decrease = _check_non_negative_float(
            min_impurity_decrease, "min_impurity_decrease"
        )
        self.max_features = _check_max_features(max_features)
        self.random_state = _check_random_state(random_state)

        self.root_: _Node | None = None
        self.n_features_: int | None = None

    def fit(self, X, y) -> DecisionTree:
        X, y = _check_Xy(X, y)
        self.n_features_ = X.shape[1]
        self._n_candidates = _resolve_max_features(self.max_features, self.n_features_)
        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow(X, y, depth=0)
        return self

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        X = _check_X(X, n_features=self.n_features_, allow_1d=True)
        p1 = np.array([self._descend(row).probability for row in X])
        return np.column_stack([1.0 - p1, p1])

    def score(self, X, y) -> float:
        X, y = _check_Xy(X, y)
        return float(np.mean(self.predict(X) == y))

    def get_depth(self) -> int:
        self._check_fitted()

        def depth_of(node: _Node) -> int:
            if node.is_leaf:
                return 0
            return 1 + max(depth_of(node.left), depth_of(node.right))

        return depth_of(self.root_)

    def _check_fitted(self) -> None:
        if self.root_ is None:
            raise RuntimeError("this DecisionTree is not fitted yet; call fit() first")

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        leaf = self._make_leaf(y)

        if y.size < self.min_samples_split:
            return leaf
        if self.max_depth is not None and depth >= self.max_depth:
            return leaf
        if np.all(y == y[0]):
            return leaf

        split = self._best_split(X, y)
        if split is None:
            return leaf

        feature, threshold = split
        mask = X[:, feature] <= threshold
        node = _Node(
            prediction=leaf.prediction,
            probability=leaf.probability,
            feature=feature,
            threshold=threshold,
        )
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    @staticmethod
    def _make_leaf(y: np.ndarray) -> _Node:
        p1 = float(np.mean(y)) if y.size else 0.0
        return _Node(prediction=int(p1 >= 0.5), probability=p1)

    def _candidate_features(self) -> np.ndarray:
        if self._n_candidates >= self.n_features_:
            return np.arange(self.n_features_)
        return self._rng.choice(self.n_features_, size=self._n_candidates, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        """Best (feature, threshold) by impurity decrease, or None if no split qualifies."""
        parent_impurity = _gini(y)
        best = None
        best_decrease = self.min_impurity_decrease

        for feature in self._candidate_features():
            candidate = _best_threshold(X[:, feature], y, parent_impurity)
            if candidate is None:
                continue

            threshold, decrease = candidate
            if decrease > best_decrease:
                best_decrease = decrease
                best = (int(feature), float(threshold))

        return best

    def _descend(self, row: np.ndarray) -> _Node:
        node = self.root_
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node
