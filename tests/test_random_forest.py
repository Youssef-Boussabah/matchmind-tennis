import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from matchmind_v1.trees import RandomForest


def test_fit_returns_self(separable):
    X, y = separable
    forest = RandomForest(n_estimators=5, random_state=0)
    assert forest.fit(X, y) is forest


def test_builds_the_requested_number_of_trees(separable):
    X, y = separable
    forest = RandomForest(n_estimators=7, random_state=0).fit(X, y)
    assert len(forest.estimators) == 7


def test_refitting_resets_the_forest(separable):
    X, y = separable
    forest = RandomForest(n_estimators=5, random_state=0).fit(X, y)
    forest.fit(X, y)
    assert len(forest.estimators) == 5


def test_same_random_state_is_reproducible(noisy_blobs):
    X, y = noisy_blobs
    first = RandomForest(n_estimators=10, max_depth=4, random_state=42).fit(X, y)
    second = RandomForest(n_estimators=10, max_depth=4, random_state=42).fit(X, y)
    assert np.array_equal(first.predict(X), second.predict(X))
    assert np.allclose(first.predict_proba(X), second.predict_proba(X))


def test_different_random_state_changes_the_forest(noisy_blobs):
    X, y = noisy_blobs
    first = RandomForest(n_estimators=10, max_depth=4, random_state=1).fit(X, y)
    second = RandomForest(n_estimators=10, max_depth=4, random_state=2).fit(X, y)
    assert not np.allclose(first.predict_proba(X), second.predict_proba(X))


def test_bootstrap_gives_trees_different_training_sets(noisy_blobs):
    X, y = noisy_blobs
    forest = RandomForest(n_estimators=5, max_depth=3, random_state=0, bootstrap=True).fit(X, y)
    thresholds = {t.root_.threshold for t in forest.estimators}
    assert len(thresholds) > 1


def test_without_bootstrap_every_tree_sees_all_rows(noisy_blobs):
    X, y = noisy_blobs
    forest = RandomForest(
        n_estimators=5, max_depth=3, max_features=None, bootstrap=False, random_state=0
    ).fit(X, y)
    predictions = [t.predict(X) for t in forest.estimators]
    assert all(np.array_equal(p, predictions[0]) for p in predictions)


def test_prediction_shape_matches_sample_count(noisy_blobs):
    X, y = noisy_blobs
    forest = RandomForest(n_estimators=5, random_state=0).fit(X, y)
    assert forest.predict(X).shape == (len(X),)


def test_predicts_a_single_sample(separable):
    X, y = separable
    forest = RandomForest(n_estimators=5, random_state=0).fit(X, y)
    assert forest.predict(X[0]).shape == (1,)
    assert forest.predict_proba(X[0]).shape == (1, 2)


def test_probabilities_are_well_formed(noisy_blobs):
    X, y = noisy_blobs
    proba = RandomForest(n_estimators=8, max_depth=4, random_state=0).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_averaged_probabilities_are_smoother_than_a_single_tree(noisy_blobs):
    X, y = noisy_blobs
    forest = RandomForest(n_estimators=15, max_depth=6, random_state=0).fit(X, y)
    proba = forest.predict_proba(X)[:, 1]
    assert np.any((proba > 0) & (proba < 1))


def test_predict_agrees_with_predict_proba(noisy_blobs):
    X, y = noisy_blobs
    forest = RandomForest(n_estimators=6, max_depth=4, random_state=0).fit(X, y)
    expected = (forest.predict_proba(X)[:, 1] >= 0.5).astype(int)
    assert np.array_equal(forest.predict(X), expected)


def test_score_matches_manual_accuracy(noisy_blobs):
    X, y = noisy_blobs
    forest = RandomForest(n_estimators=6, max_depth=4, random_state=0).fit(X, y)
    assert forest.score(X, y) == pytest.approx(np.mean(forest.predict(X) == y))


def test_learns_a_separable_problem(separable):
    X, y = separable
    forest = RandomForest(n_estimators=10, max_features=None, random_state=0).fit(X, y)
    assert forest.score(X, y) == pytest.approx(1.0)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        RandomForest().predict(np.zeros((2, 2)))


def test_rejects_invalid_n_estimators():
    with pytest.raises(ValueError, match="n_estimators"):
        RandomForest(n_estimators=0)


def test_rejects_wrong_feature_count(separable):
    X, y = separable
    forest = RandomForest(n_estimators=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="expected 2 features"):
        forest.predict(np.zeros((3, 5)))


def test_accuracy_is_comparable_to_sklearn(noisy_blobs):
    X, y = noisy_blobs
    ours = RandomForest(n_estimators=25, max_depth=6, random_state=0).fit(X, y)
    theirs = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=0).fit(X, y)
    assert ours.score(X, y) >= theirs.score(X, y) - 0.05


def test_forest_generalises_at_least_as_well_as_one_tree(noisy_blobs):
    X, y = noisy_blobs
    X_train, y_train = X[:100], y[:100]
    X_test, y_test = X[100:], y[100:]
    forest = RandomForest(n_estimators=25, max_depth=6, random_state=0).fit(X_train, y_train)
    single = RandomForest(n_estimators=1, max_depth=6, bootstrap=False, random_state=0).fit(
        X_train, y_train
    )
    assert forest.score(X_test, y_test) >= single.score(X_test, y_test)


def test_rejects_fractional_labels():
    X = np.arange(8, dtype=float).reshape(4, 2)
    with pytest.raises(ValueError, match="binary labels"):
        RandomForest(n_estimators=3, random_state=0).fit(X, np.array([0.2, 0.8, 0.2, 0.8]))


def test_fit_requires_2d_x():
    with pytest.raises(ValueError, match="2-dimensional"):
        RandomForest(n_estimators=3, random_state=0).fit(np.array([1.0, 2.0]), np.array([1]))


def test_rejects_non_integer_n_estimators():
    with pytest.raises(ValueError, match="n_estimators"):
        RandomForest(n_estimators=3.5)
    with pytest.raises(ValueError, match="n_estimators"):
        RandomForest(n_estimators=True)


def test_rejects_non_bool_bootstrap():
    with pytest.raises(ValueError, match="bootstrap"):
        RandomForest(bootstrap="yes")


def test_rejects_invalid_forwarded_tree_parameters():
    with pytest.raises(ValueError, match="max_depth"):
        RandomForest(max_depth=2.5)
    with pytest.raises(ValueError, match="max_features"):
        RandomForest(max_features="all")


def test_rejects_complex_features():
    X = np.array([[0 + 1j, 1], [1, 2], [2, 3], [3, 4]])
    with pytest.raises(ValueError, match="real-valued features"):
        RandomForest(n_estimators=3, random_state=0).fit(X, np.array([0, 1, 0, 1]))


def test_rejects_complex_labels():
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0 + 1j, 1 + 0j, 0 + 1j, 1 + 0j])
    with pytest.raises(ValueError, match="real-valued labels"):
        RandomForest(n_estimators=3, random_state=0).fit(X, y)


def test_rejects_negative_random_state():
    with pytest.raises(ValueError, match="random_state must be non-negative"):
        RandomForest(random_state=-1)
