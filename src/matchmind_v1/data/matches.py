"""Reading the ATP source files and turning the matches into canonical rows.

The raw files are winner/loser oriented and are not reliably chronological, even within a
season: individual year files step backwards in tournament date. What comes out of here is
globally ordered by date and neutralised into player 1 / player 2, so that replaying it
forward reproduces history instead of leaking it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..tennis import MAX_HEIGHT_CM, MIN_HEIGHT_CM

# The seasons the yearly files cover. 1991 is the first year the source records serve
# statistics at all — before it every row is missing them and would be dropped — and 2026
# is the last, partial at this snapshot: it stops during Roland Garros, at tourney_date
# 20260525. See data/SOURCE_SNAPSHOT.txt.
FIRST_YEAR = 1991
LAST_YEAR = 2026

TARGET_COLUMN = "RESULT"

# Match-level columns kept through the whole pipeline. best_of and draw_size are model
# inputs; the rest identify and order the match.
MATCH_COLUMNS = (
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_date",
    "match_num",
    "round",
    "best_of",
)

# Raw column suffix -> the name the field takes once the players are neutralised. The
# static fields use winner_/loser_ in the source files, the serve counts use w_/l_.
STATIC_FIELDS = {
    "id": "id",
    "ht": "height",
    "age": "age",
    "rank": "atp_rank",
    "rank_points": "atp_points",
}
SERVE_FIELDS = {
    "ace": "ace",
    "df": "double_fault",
    "svpt": "service_points",
    "1stIn": "first_serve_in",
    "1stWon": "first_serve_won",
    "2ndWon": "second_serve_won",
    "bpSaved": "break_points_saved",
    "bpFaced": "break_points_faced",
}

STATIC_COLUMNS = tuple(
    f"{side}_{suffix}" for side in ("winner", "loser") for suffix in STATIC_FIELDS
)
SERVE_COLUMNS = tuple(f"{side}_{suffix}" for side in ("w", "l") for suffix in SERVE_FIELDS)

# Everything a canonical match row needs. A row missing any of it is dropped, never imputed.
REQUIRED_COLUMNS = MATCH_COLUMNS + STATIC_COLUMNS + SERVE_COLUMNS

ID_COLUMNS = ("winner_id", "loser_id")
TEXT_COLUMNS = ("tourney_id", "tourney_name", "surface", "round")
INTEGER_COLUMNS = ("draw_size", "tourney_date", "match_num", "best_of") + ID_COLUMNS + SERVE_COLUMNS
FLOAT_COLUMNS = ("winner_ht", "winner_age", "winner_rank", "winner_rank_points",
                 "loser_ht", "loser_age", "loser_rank", "loser_rank_points")

# Groups of required fields, for reporting where the dropped rows go. A row can be
# missing fields from several groups at once, so these counts overlap and do not sum.
FIELD_GROUPS = {
    "player identity": ID_COLUMNS,
    "age and height": ("winner_age", "winner_ht", "loser_age", "loser_ht"),
    "ranking": ("winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"),
    "match context": ("surface", "best_of", "draw_size"),
    "serve statistics": SERVE_COLUMNS,
    "ordering": ("tourney_date", "tourney_id", "match_num"),
}

# Which file a row came from and where in it. Not model inputs: they break ordering
# ties and make it possible to trace a canonical row back to its source line.
SOURCE_COLUMNS = ("source_year", "source_row")

# The chronology is the first three; the source columns only break ties, so the order
# never falls back on however the yearly files happened to concatenate.
ORDER_COLUMNS = ("tourney_date", "tourney_id", "match_num") + SOURCE_COLUMNS


def load_matches(
    input_dir: str | Path, start_year: int = FIRST_YEAR, end_year: int = LAST_YEAR
) -> pd.DataFrame:
    """Read atp_matches_YYYY.csv for every year in the range, oldest first.

    Every year in the range has to be there. A missing one would quietly shorten the
    history the state is replayed over, so it is an error rather than a warning.
    """
    if start_year > end_year:
        raise ValueError(f"start year {start_year} is after end year {end_year}")

    input_dir = Path(input_dir)
    frames = []
    for year in range(start_year, end_year + 1):
        path = input_dir / f"atp_matches_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"no match file for {year}: {path}")
        year_matches = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
        year_matches["source_year"] = year
        year_matches["source_row"] = np.arange(len(year_matches))
        frames.append(year_matches)

    return pd.concat(frames, ignore_index=True)


def load_player_names(path: str | Path) -> dict[int, str]:
    """Display names from the ATP player reference file, keyed by player id.

    Names only. Height, age, rank and ranking points all come from the match rows,
    where they are recorded as of the match rather than as of today.
    """
    players = pd.read_csv(path, usecols=["player_id", "name_first", "name_last"])
    players = players.dropna(subset=["player_id"])
    parts = players[["name_first", "name_last"]].fillna("")
    names = (parts["name_first"] + " " + parts["name_last"]).str.replace(r"\s+", " ", regex=True)
    return {
        int(player_id): name
        for player_id, name in zip(players["player_id"], names.str.strip(), strict=True)
        if name
    }


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows the feature pipeline can use as they stand."""
    matches = matches.copy()
    for column in INTEGER_COLUMNS + FLOAT_COLUMNS:
        matches[column] = pd.to_numeric(matches[column], errors="coerce")
    for column in TEXT_COLUMNS:
        matches[column] = matches[column].astype("string")

    matches = matches[_is_calendar_date(matches["tourney_date"])]
    matches = matches.dropna(subset=list(REQUIRED_COLUMNS))
    matches = matches[_whole_numbers(matches[list(INTEGER_COLUMNS)])]
    matches = matches.astype({column: int for column in INTEGER_COLUMNS})

    return matches[~invalid_rows(matches)].reset_index(drop=True)


def invalid_rows(matches: pd.DataFrame) -> pd.Series:
    """Rows with impossible serve counts or heights, or the same player on both sides."""
    impossible = matches["winner_id"] == matches["loser_id"]
    for side in ("w", "l"):
        count = {name: matches[f"{side}_{suffix}"] for suffix, name in SERVE_FIELDS.items()}
        for values in count.values():
            impossible |= values < 0
        # Every point starts with a first serve. If it misses, the second serve is either
        # won, lost in play, or double faulted, so wins and double faults together cannot
        # outnumber the second serves played.
        second_serves = count["service_points"] - count["first_serve_in"]
        impossible |= count["ace"] > count["service_points"]
        impossible |= count["double_fault"] > count["service_points"]
        impossible |= count["first_serve_in"] > count["service_points"]
        impossible |= count["first_serve_won"] > count["first_serve_in"]
        impossible |= count["second_serve_won"] > second_serves
        impossible |= count["second_serve_won"] + count["double_fault"] > second_serves
        impossible |= count["ace"] > count["first_serve_won"] + count["second_serve_won"]
        impossible |= count["break_points_saved"] > count["break_points_faced"]
        impossible |= count["break_points_faced"] > count["service_points"]
    for side in ("winner", "loser"):
        height = matches[f"{side}_ht"]
        impossible |= (height < MIN_HEIGHT_CM) | (height > MAX_HEIGHT_CM)
    return impossible


def order_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Sort into one global chronology.

    `tourney_date` is the day the tournament started, not a per-match timestamp, so
    inside an event the within-event key used here is `match_num`. The files also carry
    `round`, which is kept as metadata but is not a sort key; `source_year` and
    `source_row` break whatever ties remain. The result is a deterministic total replay
    order rather than an exact played-at chronology.
    """
    return matches.sort_values(list(ORDER_COLUMNS), kind="stable").reset_index(drop=True)


def assign_players(matches: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Put each match's winner on either side, as player 1 or player 2.

    Raw rows always list the winner first, so a model trained on them would learn the
    column layout rather than the tennis. The coin flips come from a seeded generator
    and are drawn in row order, so the same rows in the same order with the same seed
    always give the same assignment. Row order itself is left alone.
    """
    rng = np.random.default_rng(seed)
    player1_won = rng.random(len(matches)) < 0.5

    assigned = matches[list(MATCH_COLUMNS + SOURCE_COLUMNS)].copy()
    for winner_prefix, loser_prefix, fields in (
        ("winner", "loser", STATIC_FIELDS),
        ("w", "l", SERVE_FIELDS),
    ):
        for suffix, name in fields.items():
            winner = matches[f"{winner_prefix}_{suffix}"].to_numpy()
            loser = matches[f"{loser_prefix}_{suffix}"].to_numpy()
            assigned[f"player1_{name}"] = np.where(player1_won, winner, loser)
            assigned[f"player2_{name}"] = np.where(player1_won, loser, winner)
    assigned[TARGET_COLUMN] = player1_won.astype(int)
    return assigned


def prepare_matches(matches: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Clean, order chronologically, then neutralise the player slots."""
    return assign_players(order_matches(clean_matches(matches)), seed=seed)


def retention_by_year(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """How many rows each year contributed and how many survived cleaning."""
    raw_rows = raw.groupby("source_year").size()
    retained = cleaned.groupby("source_year").size().reindex(raw_rows.index, fill_value=0)
    return pd.DataFrame(
        {
            "raw_rows": raw_rows,
            "retained_rows": retained,
            "removed_rows": raw_rows - retained,
            "retention_percent": (100.0 * retained / raw_rows).round(2),
        }
    ).rename_axis("year")


def missing_by_group(matches: pd.DataFrame) -> pd.Series:
    """Rows missing at least one field from each group. The groups overlap."""
    return pd.Series(
        {
            group: int(matches[list(columns)].isna().any(axis=1).sum())
            for group, columns in FIELD_GROUPS.items()
        }
    )


def duplicate_identities(matches: pd.DataFrame) -> pd.DataFrame:
    """Rows sharing a (tourney_date, tourney_id, match_num) identity with another row."""
    key = ["tourney_date", "tourney_id", "match_num"]
    return matches[matches.duplicated(subset=key, keep=False)].sort_values(key)


def backward_date_steps(dates: pd.Series) -> int:
    """How often the date goes backwards from one row to the next."""
    return int((np.diff(dates.to_numpy()) < 0).sum())


def rows_after_a_later_date(dates: pd.Series) -> int:
    """How many rows have a match played later than them somewhere above."""
    values = dates.to_numpy()
    return int((values < np.maximum.accumulate(values)).sum())


def _whole_numbers(counts: pd.DataFrame) -> pd.Series:
    """True where every value is a whole number, so casting to int discards nothing.

    A serve count of 100.9 is not a count that happens to be written as a float; it is
    a corrupt value, and truncating it to 100 would repair data the pipeline is meant
    to exclude.
    """
    return (counts % 1 == 0).all(axis=1)


def _is_calendar_date(dates: pd.Series) -> pd.Series:
    """True where the value is a real eight-digit YYYYMMDD date."""
    return pd.to_datetime(dates, format="%Y%m%d", errors="coerce").notna()
