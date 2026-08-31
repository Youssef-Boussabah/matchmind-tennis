"""Predicting an arbitrary matchup from a trained model and saved history."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tennis import (
    MatchContext,
    PlayerSnapshot,
    TennisState,
    build_features,
    feature_vector,
)
from .trees import RandomForest


def normalise_name(name: str) -> str:
    """Fold case and collapse whitespace, so ' Jannik  Sinner ' matches 'jannik sinner'."""
    return " ".join(str(name).split()).casefold()


def normalise_surface(surface: str) -> str:
    """Accept 'hard' or 'HARD' for 'Hard'. Anything else is left alone to be rejected."""
    return str(surface).strip().title()


@dataclass(frozen=True)
class PlayerProfile:
    """A player's last observed attributes in the historical data."""

    snapshot: PlayerSnapshot
    last_seen_date: int
    name: str | None = None

    @property
    def player_id(self) -> int:
        return self.snapshot.player_id

    @property
    def label(self) -> str:
        """The name if the reference data had one, otherwise the numeric id."""
        return self.name or str(self.player_id)


@dataclass(frozen=True)
class Prediction:
    """One matchup, as the model sees it. Probabilities are uncalibrated."""

    player_a: PlayerProfile
    player_b: PlayerProfile
    probability_a: float
    probability_b: float
    surface: str
    best_of: int
    draw_size: int
    history_end_date: int

    @property
    def winner(self) -> PlayerProfile:
        return self.player_a if self.probability_a >= self.probability_b else self.player_b


@dataclass
class PredictionBundle:
    """A fitted forest, the history it was fitted through, and who the players are."""

    model: RandomForest
    state: TennisState
    profiles: dict[int, PlayerProfile]
    metadata: dict
    _by_name: dict[str, list[int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for profile in self.profiles.values():
            if profile.name:
                self._by_name.setdefault(normalise_name(profile.name), []).append(profile.player_id)
        for candidates in self._by_name.values():
            candidates.sort()

    @property
    def history_end_date(self) -> int:
        return int(self.metadata["history_end_date"])

    def resolve(self, player: int | str) -> PlayerProfile:
        """Find a player by numeric ATP id or by exact full name."""
        player_id = _as_player_id(player)
        if player_id is not None:
            if player_id not in self.profiles:
                raise ValueError(f"no player with id {player_id} in this bundle")
            return self.profiles[player_id]

        key = normalise_name(player)
        if not key:
            raise ValueError("give a player as a name or a numeric ATP id")

        candidates = self._by_name.get(key, [])
        if not candidates:
            raise ValueError(f"no player named {player!r} in this bundle; try the numeric ATP id")
        if len(candidates) > 1:
            ids = ", ".join(str(i) for i in candidates)
            raise ValueError(f"{player!r} matches several players ({ids}); use the numeric ATP id")
        return self.profiles[candidates[0]]

    def predict(
        self,
        player_a: int | str,
        player_b: int | str,
        surface: str = "Hard",
        best_of: int = 3,
        draw_size: int = 128,
    ) -> Prediction:
        """Model probability that each player wins, given the saved history.

        The two players are fed to the model in a fixed order — lower ATP id first — so
        either query order uses the same underlying forest evaluation rather than scoring
        two orientations. The pair of probabilities returned here are complements.
        """
        profile_a = self.resolve(player_a)
        profile_b = self.resolve(player_b)
        if profile_a.player_id == profile_b.player_id:
            raise ValueError(f"{profile_a.label} cannot play themselves")

        context = MatchContext(normalise_surface(surface), best_of, draw_size)
        a_is_first = profile_a.player_id < profile_b.player_id
        first, second = (profile_a, profile_b) if a_is_first else (profile_b, profile_a)

        features = build_features(first.snapshot, second.snapshot, context, self.state)
        probability_first = float(
            self.model.predict_proba(feature_vector(features).reshape(1, -1))[0, 1]
        )

        probability_a = probability_first if a_is_first else 1.0 - probability_first
        return Prediction(
            player_a=profile_a,
            player_b=profile_b,
            probability_a=probability_a,
            probability_b=1.0 - probability_a,
            surface=context.surface,
            best_of=context.best_of,
            draw_size=context.draw_size,
            history_end_date=self.history_end_date,
        )


def _as_player_id(player) -> int | None:
    """The numeric id a query names, or None if the query is not a number."""
    if isinstance(player, bool):
        return None
    if isinstance(player, (int, np.integer)):
        return int(player)
    text = str(player).strip()
    return int(text) if text.isdigit() else None
