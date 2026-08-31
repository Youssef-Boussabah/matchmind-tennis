"""Historical player state, updated one completed match at a time."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .elo import DEFAULT_ELO, update_ratings

# Rolling windows the feature engine reports on. State keeps just enough history
# for the largest of them.
WINDOWS = (3, 5, 10, 25, 50, 100, 200)
MAX_HISTORY = max(WINDOWS)

SERVE_STATS = (
    "ace",
    "double_fault",
    "first_serve_in",
    "first_serve_won",
    "second_serve_won",
    "break_points_saved",
)


def _is_integer(value) -> bool:
    """True for a Python or NumPy integer. Booleans are ints to Python, but not counts."""
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


@dataclass
class PlayerMatchStats:
    """Serve statistics recorded for one player in a completed match.

    These are raw counts, and they have to add up: the rates derived from them are
    meaningless if, say, more first serves went in than were served at all.
    """

    ace: int
    double_fault: int
    service_points: int
    first_serve_in: int
    first_serve_won: int
    second_serve_won: int
    break_points_saved: int
    break_points_faced: int

    def __post_init__(self) -> None:
        for name, count in vars(self).items():
            if not _is_integer(count):
                raise ValueError(f"{name} must be a whole count, got {count!r}")

        negative = sorted(name for name, count in vars(self).items() if count < 0)
        if negative:
            raise ValueError(f"serve counts cannot be negative: {', '.join(negative)}")

        second_serves = self.service_points - self.first_serve_in
        checks = (
            (self.ace <= self.service_points, "aces cannot exceed service points"),
            (
                self.double_fault <= self.service_points,
                "double faults cannot exceed service points",
            ),
            (
                self.first_serve_in <= self.service_points,
                "first serves in cannot exceed service points",
            ),
            (
                self.first_serve_won <= self.first_serve_in,
                "first serves won cannot exceed first serves in",
            ),
            (
                self.second_serve_won <= second_serves,
                "second serves won cannot exceed second serves played",
            ),
            (
                self.second_serve_won + self.double_fault <= second_serves,
                "second serves won plus double faults cannot exceed second serves played",
            ),
            (
                self.ace <= self.first_serve_won + self.second_serve_won,
                "aces cannot exceed the service points won",
            ),
            (
                self.break_points_saved <= self.break_points_faced,
                "break points saved cannot exceed break points faced",
            ),
            (
                self.break_points_faced <= self.service_points,
                "break points faced cannot exceed service points",
            ),
        )
        for holds, message in checks:
            if not holds:
                raise ValueError(f"{message}: {self}")


@dataclass
class CompletedMatch:
    """A finished match. `result` is 1 when player 1 won, 0 when player 2 won."""

    player1_id: int
    player2_id: int
    surface: str
    result: int
    player1_stats: PlayerMatchStats
    player2_stats: PlayerMatchStats

    def __post_init__(self) -> None:
        if self.player1_id == self.player2_id:
            raise ValueError(f"a match needs two players, got {self.player1_id} on both sides")
        if not _is_integer(self.result) or self.result not in (0, 1):
            raise ValueError(f"result must be the integer 0 or 1, got {self.result!r}")


def _serve_rates(stats: PlayerMatchStats) -> dict[str, float]:
    """Serve percentages on a 0-100 scale, omitting any with an empty denominator."""
    rates = {}
    if stats.service_points > 0:
        rates["ace"] = 100.0 * stats.ace / stats.service_points
        rates["double_fault"] = 100.0 * stats.double_fault / stats.service_points
        rates["first_serve_in"] = 100.0 * stats.first_serve_in / stats.service_points
    if stats.first_serve_in > 0:
        rates["first_serve_won"] = 100.0 * stats.first_serve_won / stats.first_serve_in
    second_serves = stats.service_points - stats.first_serve_in
    if second_serves > 0:
        rates["second_serve_won"] = 100.0 * stats.second_serve_won / second_serves
    if stats.break_points_faced > 0:
        rates["break_points_saved"] = 100.0 * stats.break_points_saved / stats.break_points_faced
    return rates


@dataclass
class TennisState:
    """Everything known about players from the matches applied so far.

    Reads are pure: asking about a player who has never appeared returns a default
    and leaves the state untouched.
    """

    elo: dict[int, float] = field(default_factory=dict)
    surface_elo: dict[str, dict[int, float]] = field(default_factory=dict)
    elo_history: dict[int, deque[float]] = field(default_factory=dict)
    recent_results: dict[int, deque[int]] = field(default_factory=dict)
    serve_history: dict[int, dict[str, deque[float]]] = field(default_factory=dict)
    matches_played: dict[int, int] = field(default_factory=dict)
    h2h: dict[tuple[int, int], int] = field(default_factory=dict)
    surface_h2h: dict[str, dict[tuple[int, int], int]] = field(default_factory=dict)

    def elo_of(self, player_id: int) -> float:
        return self.elo.get(player_id, DEFAULT_ELO)

    def surface_elo_of(self, surface: str, player_id: int) -> float:
        return self.surface_elo.get(surface, {}).get(player_id, DEFAULT_ELO)

    def matches_played_of(self, player_id: int) -> int:
        return self.matches_played.get(player_id, 0)

    def h2h_wins(self, player_id: int, opponent_id: int) -> int:
        return self.h2h.get((player_id, opponent_id), 0)

    def surface_h2h_wins(self, surface: str, player_id: int, opponent_id: int) -> int:
        return self.surface_h2h.get(surface, {}).get((player_id, opponent_id), 0)

    def elo_history_of(self, player_id: int) -> Sequence[float]:
        return self.elo_history.get(player_id, ())

    def recent_results_of(self, player_id: int) -> Sequence[int]:
        return self.recent_results.get(player_id, ())

    def serve_history_of(self, player_id: int, stat: str) -> Sequence[float]:
        return self.serve_history.get(player_id, {}).get(stat, ())

    def update(self, match: CompletedMatch) -> None:
        """Apply one completed match. Callers are responsible for chronological order."""
        if match.result == 1:
            winner_id, loser_id = match.player1_id, match.player2_id
            winner_stats, loser_stats = match.player1_stats, match.player2_stats
        else:
            winner_id, loser_id = match.player2_id, match.player1_id
            winner_stats, loser_stats = match.player2_stats, match.player1_stats

        surface = match.surface

        winner_elo, loser_elo = update_ratings(self.elo_of(winner_id), self.elo_of(loser_id))
        self.elo[winner_id] = winner_elo
        self.elo[loser_id] = loser_elo

        surface_ratings = self.surface_elo.setdefault(surface, {})
        surface_ratings[winner_id], surface_ratings[loser_id] = update_ratings(
            self.surface_elo_of(surface, winner_id), self.surface_elo_of(surface, loser_id)
        )

        self._append(self.elo_history, winner_id, winner_elo)
        self._append(self.elo_history, loser_id, loser_elo)

        self._append(self.recent_results, winner_id, 1)
        self._append(self.recent_results, loser_id, 0)

        self.matches_played[winner_id] = self.matches_played_of(winner_id) + 1
        self.matches_played[loser_id] = self.matches_played_of(loser_id) + 1

        self.h2h[(winner_id, loser_id)] = self.h2h_wins(winner_id, loser_id) + 1
        surface_h2h = self.surface_h2h.setdefault(surface, {})
        surface_h2h[(winner_id, loser_id)] = (
            self.surface_h2h_wins(surface, winner_id, loser_id) + 1
        )

        for player_id, stats in ((winner_id, winner_stats), (loser_id, loser_stats)):
            history = self.serve_history.setdefault(player_id, {})
            for stat, rate in _serve_rates(stats).items():
                history.setdefault(stat, deque(maxlen=MAX_HISTORY)).append(rate)

    @staticmethod
    def _append(histories: dict, player_id: int, value) -> None:
        histories.setdefault(player_id, deque(maxlen=MAX_HISTORY)).append(value)
