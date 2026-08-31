import copy

import numpy as np
import pytest

from matchmind_v1.tennis import (
    FEATURE_NAMES,
    WINDOWS,
    CompletedMatch,
    MatchContext,
    PlayerMatchStats,
    PlayerSnapshot,
    TennisState,
    build_features,
    feature_vector,
)

HARD = MatchContext(surface="Hard", best_of=5, draw_size=128)


def player(player_id, atp_points=1000.0, atp_rank=10.0, age=25.0, height=185.0):
    return PlayerSnapshot(
        player_id=player_id,
        atp_points=atp_points,
        atp_rank=atp_rank,
        age=age,
        height=height,
    )


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


def test_schema_has_sixty_seven_unique_features():
    assert len(FEATURE_NAMES) == 67
    assert len(set(FEATURE_NAMES)) == 67


def test_build_features_matches_the_schema_exactly():
    features = build_features(player(1), player(2), HARD, TennisState())
    assert set(features) == set(FEATURE_NAMES)


def test_feature_vector_follows_schema_order():
    features = build_features(player(1), player(2), HARD, TennisState())
    vector = feature_vector(features)

    assert vector.shape == (67,)
    assert vector[0] == features["BEST_OF"]
    assert vector[FEATURE_NAMES.index("ELO_DIFF")] == features["ELO_DIFF"]


def test_reading_features_does_not_touch_state():
    state = TennisState()
    before = copy.deepcopy(state)

    build_features(player(101), player(102), HARD, state)

    assert state == before
    assert state.elo == {}
    assert state.surface_elo == {}
    assert state.matches_played == {}


def test_two_unseen_players_have_no_elo_difference():
    features = build_features(player(1), player(2), HARD, TennisState())
    assert features["ELO_DIFF"] == 0.0
    assert features["ELO_SURFACE_DIFF"] == 0.0


def test_known_player_is_compared_against_the_1500_baseline():
    state = TennisState()
    state.update(match(p1=1, p2=2, result=1))

    features = build_features(player(1), player(99), HARD, state)

    assert features["ELO_DIFF"] == pytest.approx(state.elo_of(1) - 1500.0)
    assert features["ELO_DIFF"] == pytest.approx(12.0)
    assert features["ELO_SURFACE_DIFF"] == pytest.approx(12.0)


def test_surface_elo_difference_is_surface_specific():
    state = TennisState()
    state.update(match(p1=1, p2=2, surface="Clay", result=1))

    on_clay = build_features(player(1), player(99), MatchContext("Clay", 3, 32), state)
    on_grass = build_features(player(1), player(99), MatchContext("Grass", 3, 32), state)

    assert on_clay["ELO_SURFACE_DIFF"] == pytest.approx(12.0)
    assert on_grass["ELO_SURFACE_DIFF"] == 0.0
    assert on_grass["ELO_DIFF"] == pytest.approx(12.0)


def test_head_to_head_and_experience_differences():
    state = TennisState()
    state.update(match(p1=1, p2=2, result=1))
    state.update(match(p1=1, p2=2, result=1))
    state.update(match(p1=1, p2=2, surface="Clay", result=0))
    state.update(match(p1=1, p2=3, result=1))

    features = build_features(player(1), player(2), HARD, state)

    # Player 1 leads 2-1 overall, but both of those wins came on hard courts.
    assert features["H2H_DIFF"] == 1.0
    assert features["H2H_SURFACE_DIFF"] == 2.0
    assert features["N_GAMES_DIFF"] == 1.0


def test_static_differences_use_player_one_minus_player_two():
    p1 = player(1, atp_points=5000.0, atp_rank=3.0, age=28.5, height=190.0)
    p2 = player(2, atp_points=1200.0, atp_rank=40.0, age=22.0, height=183.0)

    features = build_features(p1, p2, HARD, TennisState())

    assert features["ATP_POINTS_DIFF"] == 3800.0
    assert features["ATP_RANK_DIFF"] == -37.0
    assert features["AGE_DIFF"] == pytest.approx(6.5)
    assert features["HEIGHT_DIFF"] == 7.0
    assert features["BEST_OF"] == 5.0
    assert features["DRAW_SIZE"] == 128.0


def test_incomplete_windows_are_zero():
    state = TennisState()
    for _ in range(4):
        state.update(match(p1=1, p2=2, result=1))

    features = build_features(player(1), player(2), HARD, state)

    assert features["WIN_LAST_3_DIFF"] == 3.0
    assert features["WIN_LAST_5_DIFF"] == 0.0
    assert features["WIN_LAST_200_DIFF"] == 0.0


def test_win_windows_count_wins_over_the_last_k_matches():
    state = TennisState()
    for result in (1, 1, 0, 1, 1):
        state.update(match(p1=1, p2=2, result=result))

    features = build_features(player(1), player(2), HARD, state)

    # Player 1: [1,1,0,1,1], player 2 is the mirror image.
    assert features["WIN_LAST_5_DIFF"] == 4.0 - 1.0
    assert features["WIN_LAST_3_DIFF"] == 2.0 - 1.0


def test_windows_stay_zero_when_only_one_player_has_history():
    state = TennisState()
    for _ in range(10):
        state.update(match(p1=1, p2=2, result=1))

    features = build_features(player(1), player(77), HARD, state)

    for k in WINDOWS:
        assert features[f"WIN_LAST_{k}_DIFF"] == 0.0
        assert features[f"P_ACE_LAST_{k}_DIFF"] == 0.0
        assert features[f"ELO_GRAD_LAST_{k}_DIFF"] == 0.0


def test_serve_windows_average_the_last_k_observations():
    state = TennisState()
    for aces in (6, 9, 15):
        state.update(
            match(
                p1=1,
                p2=2,
                p1_stats=stats(ace=aces, service_points=100),
                p2_stats=stats(ace=4, service_points=100),
            )
        )

    features = build_features(player(1), player(2), HARD, state)

    assert features["P_ACE_LAST_3_DIFF"] == pytest.approx(10.0 - 4.0)
    assert features["P_ACE_LAST_5_DIFF"] == 0.0


def test_serve_windows_need_a_full_window_of_recorded_rates():
    """A stat skipped for a zero denominator does not count towards its window."""
    state = TennisState()
    for _ in range(3):
        state.update(
            match(p1=1, p2=2, p1_stats=stats(break_points_faced=0, break_points_saved=0))
        )

    features = build_features(player(1), player(2), HARD, state)

    assert features["P_BP_SAVED_LAST_3_DIFF"] == 0.0
    assert features["P_ACE_LAST_3_DIFF"] == pytest.approx(0.0)


def test_elo_gradient_uses_elo_history_not_results():
    """The binary result slope and the Elo slope disagree, and Elo must win.

    Player 1 wins three then loses three, so the last three results are a flat [0, 0, 0]
    with a slope of exactly zero, while the Elo history over the same three matches is
    clearly falling. Reading the result deque here would report zero, so this pins the
    feature to the Elo series.
    """
    state = TennisState()

    # Player 1 wins three times, then loses three times: result slope is negative.
    for _ in range(3):
        state.update(match(p1=1, p2=2, result=1))
    for _ in range(3):
        state.update(match(p1=1, p2=2, result=0))

    elo_history = list(state.elo_history_of(1))
    results = list(state.recent_results_of(1))

    elo_slope = np.polyfit(np.arange(3), np.array(elo_history[-3:]), 1)[0]
    result_slope = np.polyfit(np.arange(3), np.array(results[-3:], dtype=float), 1)[0]

    features = build_features(player(1), player(2), HARD, state)
    combined = features["ELO_GRAD_LAST_3_DIFF"]

    opponent_slope = np.polyfit(np.arange(3), np.array(list(state.elo_history_of(2))[-3:]), 1)[0]
    assert combined == pytest.approx(elo_slope - opponent_slope)

    # The Elo-history slope is on a rating scale; the binary result slope is not.
    assert abs(elo_slope) > 1.0
    assert result_slope == pytest.approx(0.0)
    assert combined != pytest.approx(result_slope)


def test_elo_gradient_is_positive_for_a_rising_player():
    state = TennisState()
    for _ in range(5):
        state.update(match(p1=1, p2=2, result=1))

    features = build_features(player(1), player(2), HARD, state)

    assert features["ELO_GRAD_LAST_3_DIFF"] > 0.0
    assert features["ELO_GRAD_LAST_5_DIFF"] > 0.0


def test_difference_features_are_antisymmetric_under_a_swap():
    state = TennisState()
    state.update(match(p1=1, p2=2, result=1))
    state.update(match(p1=1, p2=2, result=1))
    state.update(match(p1=1, p2=3, result=0))

    p1 = player(1, atp_points=5000.0, atp_rank=3.0, age=28.5, height=190.0)
    p2 = player(2, atp_points=1200.0, atp_rank=40.0, age=22.0, height=183.0)

    forward = build_features(p1, p2, HARD, state)
    reverse = build_features(p2, p1, HARD, state)

    for name in (
        "AGE_DIFF",
        "HEIGHT_DIFF",
        "ATP_RANK_DIFF",
        "ATP_POINTS_DIFF",
        "ELO_DIFF",
        "ELO_SURFACE_DIFF",
        "N_GAMES_DIFF",
        "H2H_DIFF",
        "H2H_SURFACE_DIFF",
        "WIN_LAST_3_DIFF",
        "ELO_GRAD_LAST_3_DIFF",
        "P_ACE_LAST_3_DIFF",
    ):
        assert forward[name] == pytest.approx(-reverse[name]), name


def test_context_features_are_unchanged_by_a_swap():
    forward = build_features(player(1), player(2), HARD, TennisState())
    reverse = build_features(player(2), player(1), HARD, TennisState())

    assert forward["BEST_OF"] == reverse["BEST_OF"]
    assert forward["DRAW_SIZE"] == reverse["DRAW_SIZE"]


def test_current_match_is_invisible_to_its_own_features():
    state = TennisState()
    completed = match(p1=1, p2=2, result=1)

    before = build_features(player(1), player(2), HARD, state)
    assert before["ELO_DIFF"] == 0.0
    assert before["N_GAMES_DIFF"] == 0.0
    assert before["H2H_DIFF"] == 0.0
    assert state.elo == {}

    state.update(completed)

    after = build_features(player(1), player(2), HARD, state)
    assert after["ELO_DIFF"] == pytest.approx(24.0)
    assert after["H2H_DIFF"] == 1.0
    assert after["N_GAMES_DIFF"] == 0.0


@pytest.mark.parametrize("height", [15.0, 71.0, 99.9, 250.1, 0.0])
def test_snapshot_rejects_an_impossible_height(height):
    with pytest.raises(ValueError, match="height must be in centimetres"):
        player(1, height=height)


@pytest.mark.parametrize("height", [100.0, 185.0, 250.0])
def test_snapshot_accepts_a_plausible_height(height):
    assert player(1, height=height).height == height
