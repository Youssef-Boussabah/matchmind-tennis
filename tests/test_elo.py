import pytest

from matchmind_v1.tennis.elo import DEFAULT_ELO, ELO_K, expected_score, update_ratings


def test_equal_ratings_expect_a_coin_flip():
    assert expected_score(DEFAULT_ELO, DEFAULT_ELO) == pytest.approx(0.5)


def test_expected_scores_sum_to_one():
    assert expected_score(1600, 1400) + expected_score(1400, 1600) == pytest.approx(1.0)


def test_higher_rating_is_favoured():
    assert expected_score(1700, 1500) > 0.5
    assert expected_score(1300, 1500) < 0.5


def test_update_from_equal_ratings():
    winner, loser = update_ratings(DEFAULT_ELO, DEFAULT_ELO)
    assert winner == pytest.approx(1512.0)
    assert loser == pytest.approx(1488.0)


def test_update_conserves_total_rating():
    winner, loser = update_ratings(1725.0, 1310.0)
    assert winner + loser == pytest.approx(1725.0 + 1310.0)


def test_beating_a_stronger_opponent_gains_more():
    upset_gain = update_ratings(1400, 1800)[0] - 1400
    expected_gain = update_ratings(1800, 1400)[0] - 1800
    assert upset_gain > expected_gain


def test_gain_is_bounded_by_k():
    winner, _ = update_ratings(1000.0, 2500.0)
    assert winner - 1000.0 < ELO_K
