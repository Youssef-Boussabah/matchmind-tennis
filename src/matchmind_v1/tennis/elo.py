"""Elo ratings for tennis results."""

DEFAULT_ELO = 1500.0
ELO_K = 24.0


def expected_score(rating: float, opponent_rating: float) -> float:
    """Probability that a player rated `rating` beats one rated `opponent_rating`."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def update_ratings(
    winner_rating: float, loser_rating: float, k: float = ELO_K
) -> tuple[float, float]:
    """Ratings after a completed match, as (winner, loser)."""
    shift = k * (1.0 - expected_score(winner_rating, loser_rating))
    return winner_rating + shift, loser_rating - shift
