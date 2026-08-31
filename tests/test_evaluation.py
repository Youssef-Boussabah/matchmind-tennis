import numpy as np
import pandas as pd
import pytest

from matchmind_v1.data import DATASET_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN
from matchmind_v1.evaluation import (
    TEST_PERIOD,
    TRAIN_PERIOD,
    VALIDATION_PERIOD,
    evaluate,
    features_and_target,
    majority_label,
    probability_scores,
    rank_heuristic,
    split_by_period,
)
from matchmind_v1.tennis import FEATURE_NAMES

# One date on each side of every period boundary, plus two in the partial 2026 season
# that no period is meant to reach.
DATES = [
    19910107,  # train, first season
    20001204,  # train
    20211231,  # train, last day of the period
    20220101,  # validation, first day
    20231231,  # validation, last day
    20240101,  # test, first day
    20241218,  # test
    20251231,  # test, last day
    20260101,  # excluded: partial season
    20260517,  # excluded: last date in the canonical history
]
TRAIN_DATES = [19910107, 20001204, 20211231]
VALIDATION_DATES = [20220101, 20231231]
TEST_DATES = [20240101, 20241218, 20251231]
EXCLUDED_DATES = [20260101, 20260517]


def canonical_frame(dates=DATES):
    """A dataset with the canonical columns, filled with recognisable values."""
    n = len(dates)
    frame = pd.DataFrame(
        {
            "tourney_date": dates,
            "tourney_id": [f"{d // 10000}-{i:03d}" for i, d in enumerate(dates)],
            "match_num": range(n),
            "tourney_name": "Test Open",
            "surface": "Hard",
            "round": "R32",
            "player1_id": range(100, 100 + n),
            "player2_id": range(200, 200 + n),
        }
    )
    # Each feature column sits around its own position in FEATURE_NAMES, so a
    # reordering or a substituted column is visible in the values themselves.
    for position, name in enumerate(FEATURE_NAMES):
        frame[name] = np.arange(n) / 100.0 + position
    frame[TARGET_COLUMN] = [i % 2 for i in range(n)]
    return frame[list(DATASET_COLUMNS)]


def test_the_periods_are_the_frozen_window():
    """The window was fixed before any refreshed metric was seen. Pin it exactly."""
    assert TRAIN_PERIOD == (19910101, 20211231)
    assert VALIDATION_PERIOD == (20220101, 20231231)
    assert TEST_PERIOD == (20240101, 20251231)


def test_periods_are_contiguous_and_ordered():
    assert TRAIN_PERIOD[1] < VALIDATION_PERIOD[0]
    assert VALIDATION_PERIOD[1] < TEST_PERIOD[0]
    # Contiguous across the new year: each period resumes on the 1st of January after
    # the previous one ends on the 31st of December.
    assert TRAIN_PERIOD[1] // 10000 + 1 == VALIDATION_PERIOD[0] // 10000
    assert VALIDATION_PERIOD[1] // 10000 + 1 == TEST_PERIOD[0] // 10000
    for start, end in (TRAIN_PERIOD, VALIDATION_PERIOD, TEST_PERIOD):
        assert start % 10000 == 101
        assert end % 10000 == 1231


def test_split_lands_rows_in_the_right_period():
    train, validation, test = split_by_period(canonical_frame())

    assert list(train["tourney_date"]) == TRAIN_DATES
    assert list(validation["tourney_date"]) == VALIDATION_DATES
    assert list(test["tourney_date"]) == TEST_DATES


def test_split_conserves_every_eligible_row_without_overlap():
    """Every row inside the window lands in exactly one period.

    The dataset deliberately runs past the window, so conservation is over the eligible
    rows rather than over the whole frame.
    """
    frame = canonical_frame()
    train, validation, test = split_by_period(frame)

    eligible = frame[frame["tourney_date"].between(TRAIN_PERIOD[0], TEST_PERIOD[1])]
    assert len(eligible) == len(frame) - len(EXCLUDED_DATES)

    assert len(train) + len(validation) + len(test) == len(eligible)
    identities = [set(part["match_num"]) for part in (train, validation, test)]
    assert set.union(*identities) == set(eligible["match_num"])
    assert not identities[0] & identities[1]
    assert not identities[1] & identities[2]
    assert not identities[0] & identities[2]


def test_the_partial_season_reaches_no_period():
    """2026 is incomplete, so it takes no part in fitting, selection or scoring."""
    frame = canonical_frame()
    train, validation, test = split_by_period(frame)

    excluded = frame[frame["tourney_date"] > TEST_PERIOD[1]]
    assert list(excluded["tourney_date"]) == EXCLUDED_DATES

    for part in (train, validation, test):
        assert part["tourney_date"].max() <= TEST_PERIOD[1]
        assert not set(part["match_num"]) & set(excluded["match_num"])


def test_the_periods_do_not_meet():
    train, validation, test = split_by_period(canonical_frame())

    assert train["tourney_date"].max() < validation["tourney_date"].min()
    assert validation["tourney_date"].max() < test["tourney_date"].min()


@pytest.mark.parametrize(
    ("before", "after", "sizes"),
    [
        (20211231, 20220101, (4, 4, 0)),
        (20231231, 20240101, (0, 4, 4)),
    ],
)
def test_one_tournament_date_cannot_straddle_two_periods(before, after, sizes):
    """Every match on a boundary date has to land together, on one side."""
    repeated = [before] * 4 + [after] * 4
    train, validation, test = split_by_period(canonical_frame(repeated))

    assert (len(train), len(validation), len(test)) == sizes
    for part in (train, validation, test):
        assert set(part["tourney_date"]) <= {before, after}
        assert len(set(part["tourney_date"])) <= 1


def test_split_keeps_the_rows_in_order():
    train, validation, test = split_by_period(canonical_frame())

    for part in (train, validation, test):
        assert part["tourney_date"].is_monotonic_increasing


def test_model_inputs_are_exactly_the_feature_columns():
    frame = canonical_frame()
    X, y = features_and_target(frame)

    assert X.shape == (len(frame), 67)
    assert np.array_equal(X, frame[list(FEATURE_NAMES)].to_numpy(dtype=float))
    assert np.array_equal(y, frame[TARGET_COLUMN].to_numpy())


def test_metadata_and_the_target_stay_out_of_the_model_inputs():
    frame = canonical_frame()
    X, _ = features_and_target(frame)

    assert X.shape[1] == len(FEATURE_NAMES)
    for column in (*METADATA_COLUMNS, TARGET_COLUMN):
        assert column not in FEATURE_NAMES
    # Every model input is one of the feature columns and nothing else.
    assert set(FEATURE_NAMES).isdisjoint({*METADATA_COLUMNS, TARGET_COLUMN})
    assert np.array_equal(X, frame.drop(columns=[*METADATA_COLUMNS, TARGET_COLUMN]).to_numpy())


def test_feature_order_follows_the_schema():
    frame = canonical_frame()
    X, _ = features_and_target(frame)

    # Column i of the frame was built around the value i, so the order is visible.
    assert np.argsort(X.mean(axis=0)).tolist() == list(range(67))


def test_probability_scores_on_a_perfect_and_a_hopeless_prediction():
    y = np.array([0, 1, 0, 1])

    perfect = probability_scores(y, np.array([0.0, 1.0, 0.0, 1.0]))
    assert perfect["accuracy"] == 1.0
    assert perfect["brier"] == 0.0
    assert perfect["roc_auc"] == 1.0
    assert perfect["log_loss"] < 1e-9

    uncertain = probability_scores(y, np.full(4, 0.5))
    assert uncertain["log_loss"] == pytest.approx(np.log(2))
    assert uncertain["brier"] == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [np.array([0.5, 1.5]), np.array([0.5, -0.1])])
def test_probabilities_outside_the_unit_interval_are_rejected(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        probability_scores(np.array([0, 1]), bad)


def test_non_finite_probabilities_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        probability_scores(np.array([0, 1]), np.array([0.5, np.nan]))


class AlwaysSure:
    """Predicts P(player 1 wins) straight from the first feature."""

    def predict_proba(self, X):
        p1 = (X[:, 0] > 0).astype(float)
        return np.column_stack([1.0 - p1, p1])


def test_evaluate_scores_a_fitted_model_and_times_it():
    X = np.array([[-1.0], [1.0], [-2.0], [3.0]])
    y = np.array([0, 1, 0, 1])

    scores = evaluate(AlwaysSure(), X, y)

    assert scores["accuracy"] == 1.0
    assert scores["roc_auc"] == 1.0
    assert scores["predict_seconds"] >= 0.0


@pytest.mark.parametrize(
    ("labels", "expected"),
    [([1, 1, 0], 1), ([0, 0, 1], 0), ([0, 1], 1), ([0, 0, 0], 0)],
)
def test_majority_label(labels, expected):
    assert majority_label(np.array(labels, dtype=float)) == expected


def test_rank_heuristic_backs_the_better_ranked_player():
    X = np.zeros((3, len(FEATURE_NAMES)))
    X[:, FEATURE_NAMES.index("ATP_RANK_DIFF")] = [-5.0, 5.0, 0.0]

    assert list(rank_heuristic(X)) == [1, 0, 0]
