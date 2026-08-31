"""Replaying prepared matches into the canonical pre-match feature dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..tennis import (
    FEATURE_NAMES,
    CompletedMatch,
    MatchContext,
    PlayerMatchStats,
    PlayerSnapshot,
    TennisState,
    build_features,
)
from .matches import SERVE_FIELDS, TARGET_COLUMN

# Not model inputs. They identify the match, for checking the chronology and for
# looking rows up later, and are kept apart from FEATURE_NAMES for that reason.
METADATA_COLUMNS = (
    "tourney_date",
    "tourney_id",
    "match_num",
    "tourney_name",
    "surface",
    "round",
    "player1_id",
    "player2_id",
)

DATASET_COLUMNS = METADATA_COLUMNS + FEATURE_NAMES + (TARGET_COLUMN,)


def build_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    """Build one pre-match feature row per match, in the order the matches are given.

    Each match is read against the state built from every match before it, and only
    then applied to that state. Nothing a match reveals about itself can reach its own
    feature row.
    """
    state = TennisState()
    rows = []
    for match in matches.itertuples(index=False):
        context = MatchContext(
            surface=match.surface, best_of=int(match.best_of), draw_size=int(match.draw_size)
        )
        features = build_features(
            _snapshot(match, "player1"), _snapshot(match, "player2"), context, state
        )
        rows.append(
            [getattr(match, column) for column in METADATA_COLUMNS]
            + [features[name] for name in FEATURE_NAMES]
            + [int(getattr(match, TARGET_COLUMN))]
        )
        state.update(_completed_match(match, context.surface))

    return pd.DataFrame(rows, columns=list(DATASET_COLUMNS))


def check_dataset(dataset: pd.DataFrame) -> None:
    """Raise if the generated dataset breaks any of the guarantees it is meant to have."""
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    if list(dataset.columns) != list(DATASET_COLUMNS):
        raise ValueError("dataset columns do not match the canonical schema")
    if dataset.columns.has_duplicates:
        raise ValueError("dataset has duplicate columns")

    features = dataset[list(FEATURE_NAMES)].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("dataset has missing or infinite feature values")
    if not dataset[TARGET_COLUMN].isin((0, 1)).all():
        raise ValueError(f"{TARGET_COLUMN} must be 0 or 1")
    if (dataset["player1_id"] == dataset["player2_id"]).any():
        raise ValueError("a match has the same player on both sides")
    if not dataset["tourney_date"].is_monotonic_increasing:
        raise ValueError("dataset is not in chronological order")


def replay_state(matches: pd.DataFrame) -> TennisState:
    """Apply every match in order, leaving the state as it stood after the last one.

    Unlike build_dataset this reads nothing back out, so it is what to call when the
    state itself is the thing wanted rather than the feature rows along the way.
    """
    state = TennisState()
    for match in matches.itertuples(index=False):
        state.update(_completed_match(match, match.surface))
    return state


def latest_snapshots(matches: pd.DataFrame) -> dict[int, tuple[PlayerSnapshot, int]]:
    """Each player's most recently observed attributes, with the date they were observed.

    Matches have to be in chronological order already: a later appearance simply
    overwrites an earlier one. A player who stopped playing in 2015 keeps their 2015
    rank and age, not a 2026 one.
    """
    latest: dict[int, tuple[PlayerSnapshot, int]] = {}
    for match in matches.itertuples(index=False):
        played_on = int(match.tourney_date)
        for prefix in ("player1", "player2"):
            snapshot = _snapshot(match, prefix)
            latest[snapshot.player_id] = (snapshot, played_on)
    return latest


def _snapshot(match, prefix: str) -> PlayerSnapshot:
    return PlayerSnapshot(
        player_id=int(getattr(match, f"{prefix}_id")),
        atp_points=float(getattr(match, f"{prefix}_atp_points")),
        atp_rank=float(getattr(match, f"{prefix}_atp_rank")),
        age=float(getattr(match, f"{prefix}_age")),
        height=float(getattr(match, f"{prefix}_height")),
    )


def _match_stats(match, prefix: str) -> PlayerMatchStats:
    return PlayerMatchStats(
        **{name: int(getattr(match, f"{prefix}_{name}")) for name in SERVE_FIELDS.values()}
    )


def _completed_match(match, surface: str) -> CompletedMatch:
    return CompletedMatch(
        player1_id=int(match.player1_id),
        player2_id=int(match.player2_id),
        surface=surface,
        result=int(getattr(match, TARGET_COLUMN)),
        player1_stats=_match_stats(match, "player1"),
        player2_stats=_match_stats(match, "player2"),
    )
