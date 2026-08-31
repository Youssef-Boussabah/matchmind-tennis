"""The optimized split search against the exhaustive one it replaced."""

import time

import numpy as np
import pytest

from matchmind_v1.trees import DecisionTree
from matchmind_v1.trees.decision_tree import _gini, _resolve_max_features


class ReferenceTree(DecisionTree):
    """Scores every candidate threshold by masking the column and re-counting.

    This is the straightforward reading of the split rule, and it is what the tree
    used to do. It is far too slow for real data, so it lives here as the thing the
    fast version has to agree with exactly.
    """

    def _best_split(self, X, y):
        parent_impurity = _gini(y)
        n = y.size
        best = None
        best_decrease = self.min_impurity_decrease

        for feature in self._candidate_features():
            values = np.unique(X[:, feature])
            if values.size < 2:
                continue

            for threshold in (values[:-1] + values[1:]) / 2.0:
                mask = X[:, feature] <= threshold
                n_left = int(mask.sum())
                if n_left == 0 or n_left == n:
                    continue

                weighted = (n_left * _gini(y[mask]) + (n - n_left) * _gini(y[~mask])) / n
                decrease = parent_impurity - weighted

                if decrease > best_decrease:
                    best_decrease = decrease
                    best = (int(feature), float(threshold))

        return best


def prepared(tree, X):
    """The state fit() sets up before the first split search."""
    tree.n_features_ = X.shape[1]
    tree._n_candidates = _resolve_max_features(tree.max_features, tree.n_features_)
    tree._rng = np.random.default_rng(tree.random_state)
    return tree


def structure(node):
    """The whole tree as nested tuples, so two trees can be compared exactly."""
    if node.is_leaf:
        return (node.prediction, node.probability)
    return (node.feature, node.threshold, structure(node.left), structure(node.right))


def continuous(seed, n=120):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] - 0.8 * X[:, 2] + rng.normal(scale=0.6, size=n) > 0).astype(int)
    return X, y


def repeated_values(seed, n=120):
    """Small integer columns, so most candidate thresholds sit on long runs of ties."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, 6)).astype(float)
    y = ((X[:, 1] + X[:, 4] + rng.integers(0, 3, size=n)) > 4).astype(int)
    return X, y


def constant_columns(seed, n=100):
    X, y = continuous(seed, n)
    X[:, 1] = 3.0
    X[:, 4] = -1.0
    return X, y


def duplicated_columns(seed, n=100):
    """Two identical columns score identically, so the tie-break decides."""
    X, y = continuous(seed, n)
    X[:, 3] = X[:, 0]
    X[:, 5] = X[:, 0]
    return X, y


def tiny(seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(9, 4)).astype(float)
    y = rng.integers(0, 2, size=9)
    return X, y


BUILDERS = [continuous, repeated_values, constant_columns, duplicated_columns, tiny]

SETTINGS = [
    {"max_depth": 3},
    {"max_depth": None, "min_samples_split": 2},
    {"max_depth": 4, "max_features": "sqrt", "random_state": 0},
    {"max_depth": 4, "max_features": 2, "random_state": 7},
    {"max_depth": None, "min_impurity_decrease": 0.01},
    {"max_depth": 5, "min_samples_split": 10, "max_features": "sqrt", "random_state": 3},
]


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("settings", SETTINGS, ids=range(len(SETTINGS)))
def test_optimized_tree_matches_the_reference_exactly(builder, settings):
    for seed in range(4):
        X, y = builder(seed)
        fast = DecisionTree(**settings).fit(X, y)
        slow = ReferenceTree(**settings).fit(X, y)

        assert structure(fast.root_) == structure(slow.root_)
        assert fast.get_depth() == slow.get_depth()
        assert np.array_equal(fast.predict(X), slow.predict(X))
        assert np.array_equal(fast.predict_proba(X), slow.predict_proba(X))


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_the_chosen_feature_and_threshold_agree(builder):
    for seed in range(6):
        X, y = builder(seed)
        settings = {"max_features": "sqrt", "random_state": seed}
        # Prepared, not fitted: _best_split needs the state fit() would have set up,
        # and both trees have to draw the same candidate features.
        fast = prepared(DecisionTree(**settings), X)
        slow = prepared(ReferenceTree(**settings), X)

        assert fast._best_split(X, y) == slow._best_split(X, y)


def test_a_constant_matrix_yields_no_split():
    X = np.ones((20, 4))
    y = np.array([0, 1] * 10)

    fast = prepared(DecisionTree(), X)
    slow = prepared(ReferenceTree(), X)

    assert fast._best_split(X, y) is None
    assert slow._best_split(X, y) is None


def test_identical_columns_resolve_to_the_same_one():
    """Columns 0, 3 and 5 are equal, so only the tie-break can separate them."""
    X, y = duplicated_columns(0)

    fast = DecisionTree(max_depth=1).fit(X, y)
    slow = ReferenceTree(max_depth=1).fit(X, y)

    assert fast.root_.feature == slow.root_.feature == 0
    assert fast.root_.threshold == slow.root_.threshold


def test_min_impurity_decrease_prunes_the_same_way():
    X, y = continuous(0)

    for cutoff in (0.0, 0.005, 0.05, 0.5):
        fast = DecisionTree(min_impurity_decrease=cutoff).fit(X, y)
        slow = ReferenceTree(min_impurity_decrease=cutoff).fit(X, y)
        assert structure(fast.root_) == structure(slow.root_)


def test_fits_a_dataset_the_exhaustive_search_could_not():
    """20k rows x 20 continuous columns.

    Every column has 20,000 distinct values, so the old search evaluated close to
    400,000 thresholds per node and took minutes on this input. The bound below is
    far looser than the fraction of a second the current search needs; it is there to
    catch a return to per-threshold masking, not to police normal timing noise.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20_000, 20))
    y = (X[:, 0] + 0.5 * X[:, 7] + rng.normal(scale=0.7, size=20_000) > 0).astype(int)

    started = time.perf_counter()
    tree = DecisionTree(max_depth=4, min_samples_split=20, random_state=0).fit(X, y)
    elapsed = time.perf_counter() - started

    assert tree.get_depth() == 4
    assert tree.score(X, y) > 0.6
    assert elapsed < 60.0
