import numpy as np
import pytest

from matchmind_v1.data import (
    assign_players,
    backward_date_steps,
    clean_matches,
    invalid_rows,
    load_matches,
    missing_by_group,
    order_matches,
    retention_by_year,
    rows_after_a_later_date,
)


def write_year(directory, year, matches):
    """Write a frame back out in the raw yearly file layout."""
    path = directory / f"atp_matches_{year}.csv"
    matches.drop(columns=["source_year", "source_row"]).to_csv(path, index=False)
    return path


def test_load_reads_every_year_in_the_range_oldest_first(tmp_path, raw_matches):
    write_year(tmp_path, 2019, raw_matches({"tourney_date": 20190107, "match_num": 7}))
    write_year(tmp_path, 2020, raw_matches({"match_num": 3}, {"match_num": 4}))

    loaded = load_matches(tmp_path, 2019, 2020)

    assert list(loaded["source_year"]) == [2019, 2020, 2020]
    assert list(loaded["source_row"]) == [0, 0, 1]
    assert list(loaded["match_num"]) == [7, 3, 4]


def test_load_refuses_a_missing_year(tmp_path, raw_matches):
    write_year(tmp_path, 2019, raw_matches({}))

    with pytest.raises(FileNotFoundError, match="2020"):
        load_matches(tmp_path, 2019, 2020)


def test_load_ignores_other_csv_files_in_the_directory(tmp_path, raw_matches):
    write_year(tmp_path, 2020, raw_matches({}))
    raw_matches({"winner_id": 999}).to_csv(tmp_path / "aus_open_2025.csv", index=False)

    loaded = load_matches(tmp_path, 2020, 2020)

    assert len(loaded) == 1
    assert 999 not in set(loaded["winner_id"])


@pytest.mark.parametrize(
    "missing", ["winner_id", "winner_age", "loser_ht", "winner_rank", "surface", "w_svpt"]
)
def test_rows_missing_a_required_field_are_dropped(raw_matches, missing):
    matches = raw_matches({"match_num": 1}, {"match_num": 2, missing: None})

    cleaned = clean_matches(matches)

    assert list(cleaned["match_num"]) == [1]


def test_dates_that_are_not_real_calendar_days_are_dropped(raw_matches):
    matches = raw_matches({"match_num": 1}, {"match_num": 2, "tourney_date": 20201345})

    assert list(clean_matches(matches)["match_num"]) == [1]


def test_missing_fields_are_reported_by_group(raw_matches):
    matches = raw_matches({}, {"match_num": 2, "winner_rank": None, "w_svpt": None})

    missing = missing_by_group(matches)

    assert missing["ranking"] == 1
    assert missing["serve statistics"] == 1
    assert missing["player identity"] == 0


@pytest.mark.parametrize(
    "broken",
    [
        {"loser_id": 1},
        {"w_bpSaved": -1},
        {"w_1stIn": 120},
        {"w_1stWon": 61},
        {"w_2ndWon": 41},
        {"l_bpSaved": 7},
        {"l_ace": 96},
        {"w_2ndWon": 39, "w_df": 2},
        {"l_2ndWon": 39, "l_df": 2},
        {"w_ace": 66},
        {"w_bpFaced": 101},
        {"winner_ht": 15.0},
        {"loser_ht": 71.0},
        {"winner_ht": 99.0},
        {"loser_ht": 251.0},
    ],
)
def test_structurally_impossible_rows_are_dropped(raw_matches, broken):
    matches = raw_matches({"match_num": 1}, {"match_num": 2} | broken)

    assert list(clean_matches(matches)["match_num"]) == [1]


@pytest.mark.parametrize(
    "fractional",
    [
        {"w_svpt": 100.9},
        {"w_ace": 10.9},
        {"winner_id": 1.9},
        {"match_num": 1.9},
        {"best_of": 3.5},
        {"draw_size": 32.5},
        {"tourney_date": 20200106.5},
    ],
)
def test_fractional_counts_are_dropped_rather_than_truncated(raw_matches, fractional):
    matches = raw_matches({"match_num": 1}, {"match_num": 2} | fractional)

    cleaned = clean_matches(matches)

    assert len(cleaned) == 1
    assert cleaned["match_num"].iloc[0] == 1


def test_whole_numbers_written_as_floats_are_kept(raw_matches):
    cleaned = clean_matches(raw_matches({"w_svpt": 100.0, "winner_id": 7.0}))

    assert cleaned["w_svpt"].iloc[0] == 100
    assert cleaned["winner_id"].iloc[0] == 7


@pytest.mark.parametrize("height", [100.0, 250.0, 175.0])
def test_heights_inside_the_plausible_range_are_kept(raw_matches, height):
    cleaned = clean_matches(raw_matches({"winner_ht": height, "loser_ht": height}))

    assert len(cleaned) == 1
    assert cleaned["winner_ht"].iloc[0] == height


def test_a_clean_row_is_not_flagged_as_impossible(raw_matches):
    assert not invalid_rows(raw_matches({})).any()


def test_counts_survive_cleaning_as_integers(raw_matches):
    cleaned = clean_matches(raw_matches({}))

    for column in ("tourney_date", "match_num", "winner_id", "w_svpt", "best_of"):
        assert cleaned[column].dtype == np.dtype("int64")


def test_ordering_is_date_then_tournament_then_match_number(raw_matches):
    matches = raw_matches(
        {"tourney_date": 20200113, "tourney_id": "2020-200", "match_num": 1},
        {"tourney_date": 20200106, "tourney_id": "2020-100", "match_num": 2},
        {"tourney_date": 20200106, "tourney_id": "2020-100", "match_num": 1},
        {"tourney_date": 20200106, "tourney_id": "2020-050", "match_num": 5},
    )

    ordered = order_matches(clean_matches(matches))

    assert list(zip(ordered["tourney_id"], ordered["match_num"], strict=True)) == [
        ("2020-050", 5),
        ("2020-100", 1),
        ("2020-100", 2),
        ("2020-200", 1),
    ]
    assert backward_date_steps(matches["tourney_date"]) == 1
    assert rows_after_a_later_date(matches["tourney_date"]) == 3
    assert backward_date_steps(ordered["tourney_date"]) == 0
    assert rows_after_a_later_date(ordered["tourney_date"]) == 0


def test_ordering_the_same_rows_twice_gives_the_same_order(raw_matches):
    matches = clean_matches(
        raw_matches(
            {"tourney_date": 20200203, "match_num": 4},
            {"tourney_date": 20200106, "match_num": 9},
            {"tourney_date": 20200106, "match_num": 2},
        )
    )

    first = order_matches(matches)
    second = order_matches(matches)

    assert first.equals(second)
    assert first.equals(order_matches(first))


def test_slot_assignment_puts_the_winner_on_both_sides(raw_matches):
    matches = clean_matches(raw_matches(*[{"match_num": n} for n in range(200)]))

    assigned = assign_players(matches, seed=0)

    assert set(assigned["RESULT"]) == {0, 1}
    assert 0.3 < assigned["RESULT"].mean() < 0.7


def test_attributes_and_serve_counts_follow_the_player(raw_matches):
    matches = clean_matches(
        raw_matches(
            *[
                {"match_num": n, "winner_ht": 200.0, "loser_ht": 170.0, "w_ace": 30, "l_ace": 1}
                for n in range(20)
            ]
        )
    )

    assigned = assign_players(matches, seed=0)

    for row in assigned.itertuples(index=False):
        winner_height = row.player1_height if row.RESULT == 1 else row.player2_height
        winner_aces = row.player1_ace if row.RESULT == 1 else row.player2_ace
        loser_height = row.player2_height if row.RESULT == 1 else row.player1_height
        loser_aces = row.player2_ace if row.RESULT == 1 else row.player1_ace
        assert (winner_height, winner_aces) == (200.0, 30)
        assert (loser_height, loser_aces) == (170.0, 1)


def test_slot_assignment_leaves_the_row_order_alone(raw_matches):
    ordered = order_matches(
        clean_matches(
            raw_matches(
                {"tourney_date": 20200203, "match_num": 1},
                {"tourney_date": 20200106, "match_num": 2},
                {"tourney_date": 20200106, "match_num": 1},
            )
        )
    )

    for seed in (0, 1, 2):
        assigned = assign_players(ordered, seed=seed)
        assert list(assigned["tourney_date"]) == list(ordered["tourney_date"])
        assert list(assigned["match_num"]) == list(ordered["match_num"])


def test_the_seed_decides_the_assignment(raw_matches):
    matches = clean_matches(raw_matches(*[{"match_num": n} for n in range(50)]))

    assert list(assign_players(matches, seed=0)["RESULT"]) == list(
        assign_players(matches, seed=0)["RESULT"]
    )
    assert list(assign_players(matches, seed=0)["RESULT"]) != list(
        assign_players(matches, seed=1)["RESULT"]
    )


def test_retention_is_reported_per_year(raw_matches):
    matches = raw_matches(
        {"tourney_date": 20190107, "match_num": 1},
        {"tourney_date": 20190107, "match_num": 2, "w_svpt": None},
        {"tourney_date": 20200106, "match_num": 1},
    )

    retention = retention_by_year(matches, clean_matches(matches))

    assert list(retention["raw_rows"]) == [2, 1]
    assert list(retention["retained_rows"]) == [1, 1]
    assert list(retention["removed_rows"]) == [1, 0]
    assert list(retention["retention_percent"]) == [50.0, 100.0]
