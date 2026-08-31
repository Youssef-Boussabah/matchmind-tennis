import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from matchmind_v1.trees import DecisionTree


def test_fit_returns_self(separable):
    X, y = separable
    tree = DecisionTree()
    assert tree.fit(X, y) is tree


def test_learns_a_separable_problem(separable):
    X, y = separable
    tree = DecisionTree().fit(X, y)
    assert np.array_equal(tree.predict(X), y)


def test_splits_on_the_informative_feature(separable):
    X, y = separable
    tree = DecisionTree().fit(X, y)
    assert tree.root_.feature == 0
    assert tree.root_.threshold == pytest.approx(4.5)


def test_predicts_a_single_sample_given_as_1d(separable):
    X, y = separable
    tree = DecisionTree().fit(X, y)
    prediction = tree.predict(X[0])
    assert prediction.shape == (1,)
    assert prediction[0] == y[0]


def test_predicts_a_single_sample_given_as_2d(separable):
    X, y = separable
    tree = DecisionTree().fit(X, y)
    assert tree.predict(X[:1]).shape == (1,)


def test_prediction_shape_matches_sample_count(separable):
    X, y = separable
    tree = DecisionTree().fit(X, y)
    assert tree.predict(X).shape == (len(X),)


def test_constant_features_produce_a_single_leaf():
    X = np.ones((20, 3))
    y = np.array([0] * 12 + [1] * 8)
    tree = DecisionTree().fit(X, y)
    assert tree.get_depth() == 0
    assert np.all(tree.predict(X) == 0)


def test_pure_target_gives_a_zero_depth_tree():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.ones(10, dtype=int)
    tree = DecisionTree().fit(X, y)
    assert tree.get_depth() == 0
    assert np.all(tree.predict(X) == 1)
    assert tree.predict_proba(X)[:, 1] == pytest.approx(1.0)


def test_sample_equal_to_threshold_goes_left():
    X = np.array([[0.0], [0.0], [2.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree().fit(X, y)
    assert tree.root_.threshold == pytest.approx(1.0)
    assert tree.predict(np.array([[1.0]]))[0] == 0


def test_max_depth_limits_growth(xor_like):
    X, y = xor_like
    assert DecisionTree(max_depth=1).fit(X, y).get_depth() == 1
    assert DecisionTree(max_depth=0).fit(X, y).get_depth() == 0


def test_max_depth_two_solves_xor(xor_like):
    X, y = xor_like
    assert DecisionTree(max_depth=2).fit(X, y).score(X, y) == pytest.approx(1.0)


def test_min_samples_split_prevents_splitting(separable):
    X, y = separable
    assert DecisionTree(min_samples_split=len(X) + 1).fit(X, y).get_depth() == 0


def test_min_impurity_decrease_prunes_weak_splits(noisy_blobs):
    X, y = noisy_blobs
    unrestricted = DecisionTree().fit(X, y)
    pruned = DecisionTree(min_impurity_decrease=0.05).fit(X, y)
    assert pruned.get_depth() < unrestricted.get_depth()


def test_large_min_impurity_decrease_gives_a_single_leaf(noisy_blobs):
    X, y = noisy_blobs
    assert DecisionTree(min_impurity_decrease=0.5).fit(X, y).get_depth() == 0


def test_max_features_is_reproducible(noisy_blobs):
    X, y = noisy_blobs
    first = DecisionTree(max_features="sqrt", random_state=3).fit(X, y)
    second = DecisionTree(max_features="sqrt", random_state=3).fit(X, y)
    assert np.array_equal(first.predict(X), second.predict(X))
    assert np.allclose(first.predict_proba(X), second.predict_proba(X))


def test_max_features_restricts_the_split_search(noisy_blobs):
    X, y = noisy_blobs
    limited = DecisionTree(max_depth=3, max_features=1, random_state=0).fit(X, y)
    full = DecisionTree(max_depth=3).fit(X, y)
    assert limited.score(X, y) < full.score(X, y)


def test_probabilities_are_well_formed(noisy_blobs):
    X, y = noisy_blobs
    proba = DecisionTree(max_depth=3).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_agrees_with_predict_proba(noisy_blobs):
    X, y = noisy_blobs
    tree = DecisionTree(max_depth=3).fit(X, y)
    assert np.array_equal(tree.predict(X), (tree.predict_proba(X)[:, 1] >= 0.5).astype(int))


def test_score_matches_manual_accuracy(noisy_blobs):
    X, y = noisy_blobs
    tree = DecisionTree(max_depth=3).fit(X, y)
    assert tree.score(X, y) == pytest.approx(np.mean(tree.predict(X) == y))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        DecisionTree().predict(np.zeros((2, 2)))


def test_rejects_non_binary_labels():
    X = np.arange(8, dtype=float).reshape(4, 2)
    with pytest.raises(ValueError, match="binary labels"):
        DecisionTree().fit(X, np.array([0, 1, 2, 1]))


def test_rejects_nan_input():
    X = np.array([[0.0, 1.0], [np.nan, 2.0]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        DecisionTree().fit(X, np.array([0, 1]))


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="samples"):
        DecisionTree().fit(np.zeros((4, 2)), np.array([0, 1]))


def test_rejects_wrong_feature_count(separable):
    X, y = separable
    tree = DecisionTree().fit(X, y)
    with pytest.raises(ValueError, match="expected 2 features"):
        tree.predict(np.zeros((3, 5)))


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="min_samples_split"):
        DecisionTree(min_samples_split=1)
    with pytest.raises(ValueError, match="max_depth"):
        DecisionTree(max_depth=-1)
    with pytest.raises(ValueError, match="max_features"):
        DecisionTree(max_features="all").fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_matches_sklearn_on_a_separable_problem(separable):
    X, y = separable
    ours = DecisionTree(max_depth=3).fit(X, y)
    theirs = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    assert ours.score(X, y) == pytest.approx(theirs.score(X, y)) == 1.0


def test_accuracy_is_close_to_sklearn_on_noisy_data(noisy_blobs):
    X, y = noisy_blobs
    ours = DecisionTree(max_depth=4).fit(X, y)
    theirs = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    assert ours.score(X, y) == pytest.approx(theirs.score(X, y), abs=0.03)


def test_rejects_fractional_labels():
    X = np.arange(8, dtype=float).reshape(4, 2)
    with pytest.raises(ValueError, match="binary labels"):
        DecisionTree().fit(X, np.array([0.2, 0.8, 0.2, 0.8]))
    with pytest.raises(ValueError, match="binary labels"):
        DecisionTree().fit(X, np.array([0.5, 1.5, 0.5, 1.5]))


def test_accepts_float_valued_binary_labels():
    X = np.arange(8, dtype=float).reshape(4, 2)
    tree = DecisionTree().fit(X, np.array([0.0, 1.0, 0.0, 1.0]))
    assert set(np.unique(tree.predict(X))).issubset({0, 1})


def test_fit_requires_2d_x():
    with pytest.raises(ValueError, match="2-dimensional"):
        DecisionTree().fit(np.array([1.0, 2.0, 3.0]), np.array([1]))


def test_fit_accepts_a_single_row_as_2d():
    tree = DecisionTree().fit(np.array([[1.0, 2.0, 3.0]]), np.array([1]))
    assert tree.n_features_ == 3


def test_predict_still_accepts_1d(separable):
    X, y = separable
    tree = DecisionTree().fit(X, y)
    assert tree.predict(np.array([7.0, 1.0])).shape == (1,)


def test_rejects_non_integer_parameters():
    with pytest.raises(ValueError, match="max_depth"):
        DecisionTree(max_depth=2.5)
    with pytest.raises(ValueError, match="max_depth"):
        DecisionTree(max_depth=True)
    with pytest.raises(ValueError, match="min_samples_split"):
        DecisionTree(min_samples_split=3.5)
    with pytest.raises(ValueError, match="random_state"):
        DecisionTree(random_state="abc")


def test_rejects_non_finite_min_impurity_decrease():
    with pytest.raises(ValueError, match="finite"):
        DecisionTree(min_impurity_decrease=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        DecisionTree(min_impurity_decrease=float("inf"))


def test_rejects_complex_features():
    X = np.array([[0 + 1j, 1], [1, 2], [2, 3], [3, 4]])
    with pytest.raises(ValueError, match="real-valued features"):
        DecisionTree().fit(X, np.array([0, 1, 0, 1]))


def test_rejects_complex_features_with_zero_imaginary_part():
    X = np.arange(8).reshape(4, 2).astype(complex)
    with pytest.raises(ValueError, match="real-valued features"):
        DecisionTree().fit(X, np.array([0, 1, 0, 1]))


def test_rejects_complex_labels():
    X = np.arange(8, dtype=float).reshape(4, 2)
    with pytest.raises(ValueError, match="real-valued labels"):
        DecisionTree().fit(X, np.array([0 + 1j, 1 + 0j, 0 + 1j, 1 + 0j]))


def test_rejects_negative_random_state():
    with pytest.raises(ValueError, match="random_state must be non-negative"):
        DecisionTree(random_state=-1)
