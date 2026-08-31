"""Pre-match feature construction.

Features describe what is known *before* a match. Nothing here mutates state, so the
result of the match being predicted can never leak into its own features.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean

import numpy as np

from .state import WINDOWS, TennisState, _is_integer

# Heights are in centimetres. The bounds are deliberately wide: they are here to catch
# a value recorded in the wrong unit or not at all, not to judge how tall a player is.
MIN_HEIGHT_CM = 100.0
MAX_HEIGHT_CM = 250.0

# The four surfaces the ATP match files record.
SURFACES = ("Hard", "Clay", "Grass", "Carpet")


def _real_number(value, name: str) -> float:
    """A finite number. Booleans and anything non-numeric are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


@dataclass
class PlayerSnapshot:
    """A player's non-historical attributes at match time."""

    player_id: int
    atp_points: float
    atp_rank: float
    age: float
    height: float

    def __post_init__(self) -> None:
        if not _is_integer(self.player_id) or self.player_id <= 0:
            raise ValueError(f"player_id must be a positive integer, got {self.player_id!r}")
        # No upper bound on age: the tour has had teenagers and forty-somethings.
        if _real_number(self.age, "age") <= 0.0:
            raise ValueError(f"age must be positive, got {self.age}")
        if _real_number(self.atp_rank, "atp_rank") < 1.0:
            raise ValueError(f"atp_rank must be at least 1, got {self.atp_rank}")
        if _real_number(self.atp_points, "atp_points") < 0.0:
            raise ValueError(f"atp_points cannot be negative, got {self.atp_points}")
        if not MIN_HEIGHT_CM <= _real_number(self.height, "height") <= MAX_HEIGHT_CM:
            raise ValueError(
                f"height must be in centimetres, between {MIN_HEIGHT_CM:.0f} and "
                f"{MAX_HEIGHT_CM:.0f}, got {self.height}"
            )


@dataclass
class MatchContext:
    """Details of the match itself, independent of either player."""

    surface: str
    best_of: int
    draw_size: int

    def __post_init__(self) -> None:
        if self.surface not in SURFACES:
            raise ValueError(f"surface must be one of {', '.join(SURFACES)}, got {self.surface!r}")
        if not _is_integer(self.best_of) or self.best_of not in (3, 5):
            raise ValueError(f"best_of must be 3 or 5, got {self.best_of!r}")
        # Draw sizes are not all powers of two: the source has 9, 10, 12, 18, 24 and 28.
        if not _is_integer(self.draw_size) or self.draw_size < 2:
            raise ValueError(f"draw_size must be an integer of at least 2, got {self.draw_size!r}")


# (feature prefix, name of the stat series held in TennisState.serve_history)
SERVE_FAMILIES = (
    ("P_ACE", "ace"),
    ("P_DF", "double_fault"),
    ("P_1ST_IN", "first_serve_in"),
    ("P_1ST_WON", "first_serve_won"),
    ("P_2ND_WON", "second_serve_won"),
    ("P_BP_SAVED", "break_points_saved"),
)


def _feature_names() -> tuple[str, ...]:
    names = [
        "BEST_OF",
        "DRAW_SIZE",
        "AGE_DIFF",
        "HEIGHT_DIFF",
        "ATP_RANK_DIFF",
        "ATP_POINTS_DIFF",
        "ELO_DIFF",
        "ELO_SURFACE_DIFF",
        "N_GAMES_DIFF",
        "H2H_DIFF",
        "H2H_SURFACE_DIFF",
    ]
    names += [f"ELO_GRAD_LAST_{k}_DIFF" for k in WINDOWS]
    names += [f"WIN_LAST_{k}_DIFF" for k in WINDOWS]
    for prefix, _ in SERVE_FAMILIES:
        names += [f"{prefix}_LAST_{k}_DIFF" for k in WINDOWS]
    return tuple(names)


FEATURE_NAMES = _feature_names()


def _slope(values: Sequence[float]) -> float:
    return float(np.polyfit(np.arange(len(values)), np.asarray(values, dtype=float), 1)[0])


def _window_diff(
    history1: Sequence[float],
    history2: Sequence[float],
    k: int,
    summarise: Callable[[Sequence[float]], float],
) -> float:
    """Compare the last k observations, or return 0.0 unless both players have k.

    Requiring a complete window keeps every LAST_k feature meaning what its name says,
    rather than quietly repeating a two-match average across all seven windows.
    """
    if len(history1) < k or len(history2) < k:
        return 0.0
    return float(summarise(list(history1)[-k:]) - summarise(list(history2)[-k:]))


def build_features(
    player1: PlayerSnapshot,
    player2: PlayerSnapshot,
    context: MatchContext,
    state: TennisState,
) -> dict[str, float]:
    """Pre-match features for player1 against player2, as differences where applicable."""
    p1, p2 = player1.player_id, player2.player_id
    surface = context.surface

    features = {
        "BEST_OF": float(context.best_of),
        "DRAW_SIZE": float(context.draw_size),
        "AGE_DIFF": float(player1.age - player2.age),
        "HEIGHT_DIFF": float(player1.height - player2.height),
        "ATP_RANK_DIFF": float(player1.atp_rank - player2.atp_rank),
        "ATP_POINTS_DIFF": float(player1.atp_points - player2.atp_points),
        "ELO_DIFF": state.elo_of(p1) - state.elo_of(p2),
        "ELO_SURFACE_DIFF": (
            state.surface_elo_of(surface, p1) - state.surface_elo_of(surface, p2)
        ),
        "N_GAMES_DIFF": float(state.matches_played_of(p1) - state.matches_played_of(p2)),
        "H2H_DIFF": float(state.h2h_wins(p1, p2) - state.h2h_wins(p2, p1)),
        "H2H_SURFACE_DIFF": float(
            state.surface_h2h_wins(surface, p1, p2) - state.surface_h2h_wins(surface, p2, p1)
        ),
    }

    elo1, elo2 = state.elo_history_of(p1), state.elo_history_of(p2)
    results1, results2 = state.recent_results_of(p1), state.recent_results_of(p2)
    for k in WINDOWS:
        features[f"ELO_GRAD_LAST_{k}_DIFF"] = _window_diff(elo1, elo2, k, _slope)
        features[f"WIN_LAST_{k}_DIFF"] = _window_diff(results1, results2, k, sum)

    for prefix, stat in SERVE_FAMILIES:
        served1 = state.serve_history_of(p1, stat)
        served2 = state.serve_history_of(p2, stat)
        for k in WINDOWS:
            features[f"{prefix}_LAST_{k}_DIFF"] = _window_diff(served1, served2, k, fmean)

    return features


def feature_vector(features: dict[str, float]) -> np.ndarray:
    """Feature values ordered by FEATURE_NAMES, ready to hand to a model."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)
