from datetime import date, timedelta

import pandas as pd
import pytest

from matchmind_v1.data import (
    DATASET_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    build_dataset,
    check_dataset,
    prepare_matches,
)
from matchmind_v1.tennis import FEATURE_NAMES

SERVE_FEATURES = [name for name in FEATURE_NAMES if name.startswith("P_")]

# One player aces 90 of 100 service points, which also means winning at least 90 of them;
# the other aces 1 of 95. Extreme, but a match that could have been played.
EXTREME_SERVING = {
    "w_ace": 90,
    "w_svpt": 100,
    "w_1stIn": 95,
    "w_1stWon": 90,
    "w_2ndWon": 3,
    "w_df": 2,
    "l_ace": 1,
    "l_svpt": 95,
}


def weekly(n):
    """The nth Monday from the start of 2020, as a YYYYMMDD integer."""
    return int((date(2020, 1, 6) + timedelta(weeks=n)).strftime("%Y%m%d"))


def series_of_matches(raw_matches, count, **overrides):
    """`count` matches between the same two players, a week apart."""
    return raw_matches(
        *[
            {"tourney_date": weekly(n), "tourney_id": f"2020-{n:03d}", "match_num": 1} | overrides
            for n in range(count)
        ]
    )


def test_schema_is_metadata_then_features_then_target():
    assert DATASET_COLUMNS == METADATA_COLUMNS + FEATURE_NAMES + (TARGET_COLUMN,)
    assert len(FEATURE_NAMES) == 67
    assert len(DATASET_COLUMNS) == len(set(DATASET_COLUMNS))
    assert not set(METADATA_COLUMNS) & set(FEATURE_NAMES)
    assert TARGET_COLUMN not in FEATURE_NAMES


def test_built_columns_follow_the_schema(raw_matches):
    dataset = build_dataset(prepare_matches(series_of_matches(raw_matches, 4)))

    assert list(dataset.columns) == list(DATASET_COLUMNS)
    assert len(dataset) == 4
    check_dataset(dataset)


def test_a_match_cannot_see_its_own_serve_statistics(raw_matches):
    """Player 1 aces 90 of 100 every match; that can only show up from the fourth row on."""
    matches = series_of_matches(raw_matches, 5, **EXTREME_SERVING)
    dataset = build_dataset(prepare_matches(matches))

    for row in range(3):
        assert dataset.loc[row, SERVE_FEATURES].abs().max() == 0.0

    assert abs(dataset.loc[3, "P_ACE_LAST_3_DIFF"]) == pytest.approx(90.0 - 100.0 / 95.0)
    assert dataset.loc[4, "P_ACE_LAST_5_DIFF"] == 0.0


def test_serve_history_follows_the_player_through_the_slot_swap(raw_matches):
    matches = series_of_matches(raw_matches, 4, **EXTREME_SERVING)
    dataset = build_dataset(prepare_matches(matches))
    fourth = dataset.loc[3]

    winner_is_player1 = fourth[TARGET_COLUMN] == 1
    ace_diff = fourth["P_ACE_LAST_3_DIFF"]

    assert (ace_diff > 0) == winner_is_player1


def test_replay_follows_the_sorted_order_not_the_input_order(raw_matches):
    reversed_input = raw_matches(
        {"tourney_date": 20200203, "tourney_id": "2020-300", "match_num": 1},
        {"tourney_date": 20200106, "tourney_id": "2020-100", "match_num": 2},
        {"tourney_date": 20200106, "tourney_id": "2020-100", "match_num": 1},
    )

    dataset = build_dataset(prepare_matches(reversed_input))

    assert list(dataset["tourney_date"]) == [20200106, 20200106, 20200203]
    assert list(dataset["match_num"]) == [1, 2, 1]
    # The first match is played from nothing; the last has both earlier results behind it.
    assert dataset.loc[0, "ELO_DIFF"] == 0.0
    assert dataset.loc[0, "H2H_DIFF"] == 0.0
    assert abs(dataset.loc[1, "H2H_DIFF"]) == 1.0
    assert abs(dataset.loc[2, "H2H_DIFF"]) == 2.0
    assert dataset.loc[2, "ELO_DIFF"] != 0.0


def test_appending_a_later_match_leaves_the_earlier_rows_alone(raw_matches):
    prefix = series_of_matches(raw_matches, 6)
    extended = pd.concat(
        [prefix, series_of_matches(raw_matches, 7).tail(1)], ignore_index=True
    )
    extended["source_row"] = range(len(extended))

    before = build_dataset(prepare_matches(prefix, seed=0))
    after = build_dataset(prepare_matches(extended, seed=0))

    assert len(after) == len(before) + 1
    pd.testing.assert_frame_equal(after.head(len(before)), before)


def test_building_twice_gives_the_same_dataset(raw_matches):
    matches = series_of_matches(raw_matches, 8)

    first = build_dataset(prepare_matches(matches, seed=0))
    second = build_dataset(prepare_matches(matches, seed=0))

    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_changes_the_sides_but_not_the_order(raw_matches):
    matches = series_of_matches(raw_matches, 8)

    first = build_dataset(prepare_matches(matches, seed=0))
    other = build_dataset(prepare_matches(matches, seed=1))

    assert list(first["tourney_date"]) == list(other["tourney_date"])
    assert list(first["match_num"]) == list(other["match_num"])
    assert list(first[TARGET_COLUMN]) != list(other[TARGET_COLUMN])
    check_dataset(other)


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: d.assign(RESULT=2),
        lambda d: d.assign(player2_id=d["player1_id"]),
        lambda d: d.assign(ELO_DIFF=float("nan")),
        lambda d: d.assign(ELO_DIFF=float("inf")),
        lambda d: d.iloc[::-1],
        lambda d: d.drop(columns=["surface"]),
    ],
)
def test_check_dataset_catches_a_broken_dataset(raw_matches, damage):
    dataset = build_dataset(prepare_matches(series_of_matches(raw_matches, 4)))

    with pytest.raises(ValueError):
        check_dataset(damage(dataset))
