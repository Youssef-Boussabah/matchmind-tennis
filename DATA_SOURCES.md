# Data Sources

Where the data in this repository comes from, and what the project generates from it.

---

## What the project reads and writes

| Role | Path | Notes |
|---|---|---|
| **Match history** | `data/all/atp_matches_1991.csv` … `atp_matches_2026.csv` (36 files) | The only match files the pipeline reads. Loaded by year, so nothing outside the requested range can be picked up by accident. |
| **Player names** | `data/atp_players.csv` | Names only, for resolving a player by name instead of numeric id. Height, age, rank and ranking points all come from the match rows, where they are recorded as of the match. |
| **Source documentation** | `data/matches_data_dictionary.txt` | Column-by-column description of the match files. |
| **Source snapshot** | `data/SOURCE_SNAPSHOT.txt` | Which mirror and commit the match files were taken from, and how far the final season runs. |
| **Data licence** | `data/LICENSE.md` | Attribution and licence terms for the third-party files above. |
| **Generated feature set** | `data/processed/match_features.csv` | Built by `scripts/build_dataset.py`. 99,158 matches, 8 metadata columns, 67 features and the target. SHA-256 `22425a372212c475ff0eacaf7a85916cc2468d16a504641ad06883754ddbe1ef`. Not committed — rebuild it. |
| **Evaluation results** | `results/validation_results.csv`, `results/test_metrics.csv` | Written by `scripts/evaluate_models.py`. Current: measured on the dataset above. |
| **Prediction bundle** | `models/matchmind_v1_bundle.json.gz` | Built by `scripts/build_prediction_bundle.py`. Holds the trained forest, the replayed player state, and one profile per player. Current: trained on all 99,158 canonical matches, 25 trees, state through `tourney_date` 20260517, 1,968 profiles. SHA-256 `2b97e793fc946bf27b1a024ab46608264d1623d5927b6db21fd43d7ce9b22d24`. |

Coverage is **1991–2026**, from the snapshot recorded in
[data/SOURCE_SNAPSHOT.txt](data/SOURCE_SNAPSHOT.txt). **2025 is a complete season; 2026 is
partial**, running to `tourney_date` 20260525 — the start of Roland Garros, which is the last
event in the files. That draw carries no serve statistics, so the last match the cleaner keeps
is a week earlier, on 20260517.

Nothing is downloaded at build or prediction time; the pipeline reads only the files already in
`data/`. The loader takes a year range, so same-schema yearly files can be added later without
changing the code.

Processing is read-only with respect to the source files: nothing rewrites or normalises a
source CSV in place.

---

## Jeff Sackmann / Tennis Abstract

The historical ATP data this project is built on comes from Jeff Sackmann's `tennis_atp`
dataset. The file schemas match that dataset directly, and `data/matches_data_dictionary.txt`
is the data dictionary distributed with it.

- Original source repository (**currently unavailable** — it returns 404 as of this writing):
  <https://github.com/JeffSackmann/tennis_atp>

Because the original repository is unreachable, the files come from a mirror of it, pinned to
one commit and recorded in [data/SOURCE_SNAPSHOT.txt](data/SOURCE_SNAPSHOT.txt):

| | |
|---|---|
| Mirror | <https://github.com/Kadantte/tennis_atp> |
| Pinned commit | `712be0c5ade693cdab9e69c23a71a0edf5a23c44` |

The mirror is a redistribution channel, not an author: the data is Jeff Sackmann's, and the
attribution and licence below are unchanged by where the bytes were fetched from.

That the mirror reproduces the dataset faithfully is not taken on trust. The 1991–2024 files
this repository already held were compared against the same pinned snapshot before 2025 and
2026 were added, and **all 34 were content-identical** — same row counts, same columns, same
values. The only difference was line endings: the snapshot uses CRLF, this repository uses LF,
and removing the carriage returns reproduced each existing file byte for byte. Those 34 files
were therefore left exactly as they were rather than recopied.

2025 and 2026 were imported from that same pinned snapshot and normalised the same way, CRLF to
LF, with no CSV value altered. No other provider is blended in, and the project contains no
download code.

| Path | Contents |
|---|---|
| `data/all/atp_matches_1991.csv` … `atp_matches_2026.csv` | Year-by-year ATP match results, 49 columns each |
| `data/atp_players.csv` | Player reference table: `player_id, name_first, name_last, hand, dob, ioc, height, wikidata_id` |
| `data/matches_data_dictionary.txt` | Column-by-column description of the match files |

These are third-party data files, retained with their attribution as part of the project's
provenance. The dataset's own README identifies the tennis data as **CC BY-NC-SA 4.0**,
attributed to Jeff Sackmann / Tennis Abstract. The notice covering the files in `data/` is in
[data/LICENSE.md](data/LICENSE.md); it applies to that data, not to this project's Python
source.

### Tennis Elo methodology

The Elo approach in the feature engineering follows the write-up published on Tennis Abstract:

- <https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/>

A related reference for five-set win probability:

- <https://github.com/JeffSackmann/tennis_misc/blob/master/fiveSetProb.py>

---

## Generated files

`data/processed/match_features.csv` and `models/matchmind_v1_bundle.json.gz` are outputs of
this project's own code, generated from the match files above. The attribution in the previous
section carries through to both.

Both are reproducible. The dataset's row values and ordering are deterministic from the pinned
inputs and the slot-assignment seed — a total sort, a seeded generator for the player slots, and
a deterministic feature and state replay — and the recorded CSV hash is reproduced in the
project's pinned environment. The bundle is written as deterministic JSON with a fixed gzip
timestamp and no stored filename, removing key-order, timestamp and filename nondeterminism;
repeated builds in that same validated environment reproduce the recorded bundle bytes.

Extending the history to 2026 did not disturb what came before it. Every match added is dated
after the previous cutoff, and the sort is total, so the rebuilt feature set begins with the
95,367 rows of the 1991–2024 dataset reproduced byte for byte and appends 3,791 new ones.

The evaluation outputs in `results/` are **current**: they were measured on this dataset, over
1991–2021 / 2022–2023 / 2024–2025 with the partial 2026 season excluded.

`models/matchmind_v1_bundle.json.gz` is also **current**. It holds the from-scratch forest — 25
trees, `max_depth=8`, `min_samples_split=20` — refitted on all 99,158 canonical matches, with
player state replayed to `tourney_date` 20260517 and 1,968 player profiles. Its SHA-256 is
`2b97e793fc946bf27b1a024ab46608264d1623d5927b6db21fd43d7ce9b22d24`.

That full-history refit is a deployment artifact, not an evaluation result: it has seen every
match it could be scored against. The held-out numbers in `results/` come from a separate model
that never saw 2024 onward.
