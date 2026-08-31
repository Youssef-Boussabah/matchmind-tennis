import numpy as np
import pytest

from matchmind_v1.tennis import CompletedMatch, PlayerMatchStats, TennisState
from matchmind_v1.tennis.elo import DEFAULT_ELO
from matchmind_v1.tennis.state import MAX_HISTORY


def stats(**overrides):
    base = dict(
        ace=10,
        double_fault=2,
        service_points=100,
        first_serve_in=60,
        first_serve_won=45,
        second_serve_won=20,
        break_points_saved=3,
        break_points_faced=5,
    )
    base.update(overrides)
    return PlayerMatchStats(**base)


def match(p1=1, p2=2, surface="Hard", result=1, p1_stats=None, p2_stats=None):
    return CompletedMatch(
        player1_id=p1,
        player2_id=p2,
        surface=surface,
        result=result,
        player1_stats=p1_stats or stats(),
        player2_stats=p2_stats or stats(),
    )


def test_unknown_players_read_as_default_without_being_inserted():
    state = TennisState()

    assert state.elo_of(99) == DEFAULT_ELO
    assert state.surface_elo_of("Clay", 99) == DEFAULT_ELO
    assert state.matches_played_of(99) == 0
    assert state.h2h_wins(99, 98) == 0
    assert state.surface_h2h_wins("Clay", 99, 98) == 0
    assert list(state.elo_history_of(99)) == []
    assert list(state.recent_results_of(99)) == []
    assert list(state.serve_history_of(99, "ace")) == []

    assert state.elo == {}
    assert state.surface_elo == {}
    assert state.matches_played == {}
    assert state.h2h == {}


def test_first_match_updates_elo():
    state = TennisState()
    state.update(match(result=1))

    assert state.elo_of(1) == pytest.approx(1512.0)
    assert state.elo_of(2) == pytest.approx(1488.0)


def test_player_two_winning_is_handled():
    state = TennisState()
    state.update(match(result=0))

    assert state.elo_of(2) == pytest.approx(1512.0)
    assert state.elo_of(1) == pytest.approx(1488.0)
    assert state.h2h_wins(2, 1) == 1
    assert state.h2h_wins(1, 2) == 0
    assert list(state.recent_results_of(2)) == [1]
    assert list(state.recent_results_of(1)) == [0]


def test_surface_elo_moves_only_on_the_surface_played():
    state = TennisState()
    state.update(match(surface="Clay"))

    assert state.surface_elo_of("Clay", 1) == pytest.approx(1512.0)
    assert state.surface_elo_of("Grass", 1) == DEFAULT_ELO
    assert state.surface_elo_of("Hard", 2) == DEFAULT_ELO


def test_counts_results_and_head_to_head():
    state = TennisState()
    state.update(match(result=1))
    state.update(match(result=1))
    state.update(match(result=0))

    assert state.matches_played_of(1) == 3
    assert state.matches_played_of(2) == 3
    assert list(state.recent_results_of(1)) == [1, 1, 0]
    assert state.h2h_wins(1, 2) == 2
    assert state.h2h_wins(2, 1) == 1
    assert state.surface_h2h_wins("Hard", 1, 2) == 2


def test_head_to_head_is_tracked_per_surface():
    state = TennisState()
    state.update(match(surface="Clay", result=1))
    state.update(match(surface="Grass", result=0))

    assert state.h2h_wins(1, 2) == 1
    assert state.surface_h2h_wins("Clay", 1, 2) == 1
    assert state.surface_h2h_wins("Grass", 1, 2) == 0
    assert state.surface_h2h_wins("Grass", 2, 1) == 1


def test_elo_history_records_ratings_not_results():
    state = TennisState()
    for _ in range(3):
        state.update(match(result=1))

    history = list(state.elo_history_of(1))
    assert len(history) == 3
    assert all(value > 1500.0 for value in history)
    assert history == sorted(history)
    assert history[0] == pytest.approx(1512.0)


def test_rolling_histories_are_capped():
    state = TennisState()
    for _ in range(MAX_HISTORY + 25):
        state.update(match())

    assert len(state.elo_history_of(1)) == MAX_HISTORY
    assert len(state.recent_results_of(1)) == MAX_HISTORY
    assert len(state.serve_history_of(1, "ace")) == MAX_HISTORY


def test_serve_rates_follow_the_player_not_the_winner_slot():
    state = TennisState()
    state.update(
        match(
            result=0,
            p1_stats=stats(ace=20, service_points=100),
            p2_stats=stats(ace=5, service_points=100),
        )
    )

    assert list(state.serve_history_of(1, "ace")) == [20.0]
    assert list(state.serve_history_of(2, "ace")) == [5.0]


def test_serve_rates_are_percentages():
    state = TennisState()
    state.update(
        match(
            p1_stats=stats(
                ace=10,
                double_fault=5,
                service_points=100,
                first_serve_in=60,
                first_serve_won=48,
                second_serve_won=20,
                break_points_saved=3,
                break_points_faced=4,
            )
        )
    )

    assert list(state.serve_history_of(1, "ace")) == [10.0]
    assert list(state.serve_history_of(1, "double_fault")) == [5.0]
    assert list(state.serve_history_of(1, "first_serve_in")) == [60.0]
    assert list(state.serve_history_of(1, "first_serve_won")) == [80.0]
    assert list(state.serve_history_of(1, "second_serve_won")) == [50.0]
    assert list(state.serve_history_of(1, "break_points_saved")) == [75.0]


def test_all_first_serves_still_records_the_other_rates():
    """A match with no second serves must not suppress ace, DF or first-serve stats."""
    state = TennisState()
    state.update(
        match(
            p1_stats=stats(
                ace=12,
                double_fault=0,
                service_points=80,
                first_serve_in=80,
                first_serve_won=60,
                second_serve_won=0,
            )
        )
    )

    assert list(state.serve_history_of(1, "ace")) == [15.0]
    assert list(state.serve_history_of(1, "double_fault")) == [0.0]
    assert list(state.serve_history_of(1, "first_serve_in")) == [100.0]
    assert list(state.serve_history_of(1, "first_serve_won")) == [75.0]
    assert list(state.serve_history_of(1, "second_serve_won")) == []


def test_zero_denominators_record_nothing_rather_than_a_placeholder():
    state = TennisState()
    state.update(
        match(
            p1_stats=stats(
                ace=0,
                double_fault=0,
                service_points=0,
                first_serve_in=0,
                first_serve_won=0,
                second_serve_won=0,
                break_points_saved=0,
                break_points_faced=0,
            )
        )
    )

    for stat in ("ace", "double_fault", "first_serve_in", "first_serve_won", "break_points_saved"):
        assert list(state.serve_history_of(1, stat)) == []


@pytest.mark.parametrize("result", [2, -1, 0.5, 0.0, 1.0, None, "1", True, False])
def test_result_must_be_the_integer_zero_or_one(result):
    with pytest.raises(ValueError, match="result must be the integer 0 or 1"):
        match(result=result)


@pytest.mark.parametrize("result", [0, 1, np.int64(0), np.int64(1)])
def test_integer_results_are_accepted(result):
    assert match(result=result).result == result


def test_a_player_cannot_play_themselves():
    with pytest.raises(ValueError, match="two players"):
        match(p1=7, p2=7)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ace": -1}, "cannot be negative"),
        ({"break_points_saved": -4, "break_points_faced": 0}, "cannot be negative"),
        ({"ace": 120}, "aces cannot exceed service points"),
        ({"double_fault": 120}, "double faults cannot exceed service points"),
        ({"first_serve_in": 130}, "first serves in cannot exceed service points"),
        ({"first_serve_won": 61}, "first serves won cannot exceed first serves in"),
        ({"second_serve_won": 41}, "second serves won cannot exceed second serves played"),
        ({"break_points_saved": 6}, "break points saved cannot exceed break points faced"),
        (
            {"second_serve_won": 39, "double_fault": 2},
            "second serves won plus double faults cannot exceed second serves played",
        ),
        ({"ace": 66}, "aces cannot exceed the service points won"),
        ({"break_points_faced": 101}, "break points faced cannot exceed service points"),
    ],
)
def test_impossible_serve_counts_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        stats(**overrides)


@pytest.mark.parametrize("count", [1.5, 10.0, True, False, None, "10"])
def test_serve_counts_must_be_whole_numbers(count):
    with pytest.raises(ValueError, match="must be a whole count"):
        stats(ace=count)


def test_a_double_fault_uses_up_a_second_serve():
    """39 second serves won and 2 double faults need 41 second serves; only 40 were played."""
    assert stats(second_serve_won=38, double_fault=2).second_serve_won == 38

    with pytest.raises(ValueError, match="plus double faults"):
        stats(second_serve_won=39, double_fault=2)


def test_numpy_integer_counts_are_accepted():
    assert stats(ace=np.int64(9)).ace == 9
