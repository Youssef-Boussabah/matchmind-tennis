"""The active source range, and the provenance record for the files it reads.

These guard the boundary between the repository's data and the pipeline: which seasons
count as active history, that the newest ones are actually picked up, and that the
snapshot the files came from is still recorded alongside them.
"""

import hashlib
from pathlib import Path

import pytest

from matchmind_v1.data import FIRST_YEAR, LAST_YEAR, clean_matches, load_matches

REPO_ROOT = Path(__file__).resolve().parents[1]
MATCH_FILES = REPO_ROOT / "data" / "all"
SNAPSHOT = REPO_ROOT / "data" / "SOURCE_SNAPSHOT.txt"
CANONICAL_DATASET = REPO_ROOT / "data" / "processed" / "match_features.csv"

# The one place the generated dataset's identity is recorded. It is rebuilt by
# scripts/build_dataset.py and not committed, so the check below skips when it is absent.
CANONICAL_DATASET_SHA256 = "22425a372212c475ff0eacaf7a85916cc2468d16a504641ad06883754ddbe1ef"
CANONICAL_DATASET_ROWS = 99158

# 2026 stops during Roland Garros. That draw carries no serve statistics, so the last
# match the cleaner keeps is a week earlier than the last match in the files.
LAST_SOURCE_DATE = 20260525
LAST_CANONICAL_DATE = 20260517


def write_year(directory, year, matches):
    """Write a frame back out in the raw yearly file layout."""
    path = directory / f"atp_matches_{year}.csv"
    matches.drop(columns=["source_year", "source_row"]).to_csv(path, index=False)
    return path


def test_the_active_range_runs_from_1991_to_2026():
    assert FIRST_YEAR == 1991
    assert LAST_YEAR == 2026


def test_every_active_season_has_a_match_file():
    missing = [
        year
        for year in range(FIRST_YEAR, LAST_YEAR + 1)
        if not (MATCH_FILES / f"atp_matches_{year}.csv").exists()
    ]
    assert missing == []


@pytest.mark.parametrize("year", [2025, 2026])
def test_the_new_seasons_are_present_and_readable(year):
    """2025 and 2026 are found by name and parse under the same schema as the rest."""
    matches = load_matches(MATCH_FILES, year, year)

    assert len(matches) > 0
    assert set(matches["source_year"]) == {year}


def test_the_defaults_read_every_active_season(tmp_path, raw_matches):
    """load_matches with no explicit range covers the whole active history.

    Written against synthetic files so it asserts the default boundaries rather than
    the size of the real data.
    """
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        write_year(tmp_path, year, raw_matches({"tourney_date": year * 10000 + 205}))

    loaded = load_matches(tmp_path)

    assert list(loaded["source_year"]) == list(range(FIRST_YEAR, LAST_YEAR + 1))


def test_the_final_partial_season_is_accepted(tmp_path, raw_matches):
    """A short final season is history, not an error: 2026 is partial by nature."""
    write_year(tmp_path, LAST_YEAR - 1, raw_matches({"tourney_date": 20250106}, {"match_num": 2}))
    write_year(tmp_path, LAST_YEAR, raw_matches({"tourney_date": 20260104}))

    loaded = load_matches(tmp_path, LAST_YEAR - 1, LAST_YEAR)

    assert list(loaded["source_year"]) == [LAST_YEAR - 1, LAST_YEAR - 1, LAST_YEAR]


def test_the_last_season_ends_where_the_snapshot_says_it_does():
    """The files stop mid-Roland-Garros, and cleaning stops a week earlier still."""
    raw = load_matches(MATCH_FILES, LAST_YEAR, LAST_YEAR)

    assert int(raw["tourney_date"].max()) == LAST_SOURCE_DATE
    assert int(clean_matches(raw)["tourney_date"].max()) == LAST_CANONICAL_DATE


def test_the_source_snapshot_records_the_pinned_provenance():
    recorded = dict(
        line.split("=", 1)
        for line in SNAPSHOT.read_text(encoding="utf-8-sig").splitlines()
        if "=" in line and not line.startswith("#")
    )

    assert recorded["mirror_commit"] == "712be0c5ade693cdab9e69c23a71a0edf5a23c44"
    assert recorded["original_dataset"] == "Jeff Sackmann / Tennis Abstract"
    assert recorded["license"] == "CC BY-NC-SA 4.0"
    assert int(recorded["min_season"]) == FIRST_YEAR
    assert int(recorded["max_season"]) == LAST_YEAR
    assert int(recorded["latest_tourney_date"]) == LAST_SOURCE_DATE
    assert recorded["partial_final_season"] == "true"


@pytest.mark.skipif(
    not CANONICAL_DATASET.exists(),
    reason="generated dataset is not committed; rebuild with scripts/build_dataset.py",
)
def test_the_generated_dataset_is_the_one_recorded_here():
    digest = hashlib.sha256()
    lines = 0
    with open(CANONICAL_DATASET, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")

    assert lines - 1 == CANONICAL_DATASET_ROWS
    assert digest.hexdigest() == CANONICAL_DATASET_SHA256
