# Architecture

MatchMind Tennis V1 is a chronological tennis-state pipeline feeding a from-scratch binary tree
ensemble. Raw ATP match files are cleaned and put into one global date order, replayed forward
through an evolving `TennisState` to produce one pre-match feature row per match, and those rows
train the decision tree and random forest in `matchmind_v1.trees`.

The shape of the system comes from one constraint: a match must never contribute to its own
feature row, and a season must never contribute to a model that is scored on it. Almost every
boundary below exists to make one of those two things structurally impossible rather than merely
avoided by convention.

Dataset construction, evaluation and deployment are three separate programs over one shared
artifact. They do not call each other; the contract between them is the canonical CSV and the
`FEATURE_NAMES` order it is written in.

---

## System map

```
data/all/atp_matches_1991.csv ... atp_matches_2026.csv
        |
        v
load_matches      read the yearly files, tag each row with its origin
        |
        v
clean_matches     drop rows the feature engine cannot use as they stand
        |
        v
order_matches     one global chronology, deterministic to the last tie
        |
        v
assign_players    winner/loser -> player1/player2 on a seeded coin flip
        |
        v
build_dataset     TennisState replay
        |
        +-------> build_features(match_t)   read state, never write
        |         state.update(match_t)     advance state, never read back
        |
        v
data/processed/match_features.csv       8 metadata + 67 features + RESULT
        |
        +-------> evaluate_models.py    chronological periods -> results/*.csv
        |
        +-------> build_prediction_bundle.py
                       |
                       |  refit on all canonical rows
                       |  replay state over the same history
                       |  one profile per player
                       v
                 models/matchmind_v1_bundle.json.gz
                       |
                       v
                 load_bundle -> PredictionBundle.predict -> scripts/predict_match.py
```

---

## Package responsibilities

The package separates six responsibilities. They do not form a single ranked stack — `artifacts.py`
and `prediction.py` are both domain-aware and sit alongside the pipeline rather than above or below
it — so the useful thing to state is the actual dependency direction of each part:

| Component | Responsibility | Knows about tennis? |
|---|---|---|
| `trees/` | binary classification on a float matrix | no — domain-agnostic |
| `tennis/` | player state and pre-match feature semantics | yes — owns the domain |
| `data/` | bridges ATP source frames into the domain objects used during replay | yes, plus the source file format |
| `evaluation.py` | chronological partitions and scoring over the canonical schema | only the feature and target column names |
| `prediction.py` | answers a matchup from frozen state and a frozen model | yes — builds features |
| `artifacts.py` | serializes and restores both the model and the tennis state | yes — must encode both |

The one boundary that carries real weight is the first: the models cannot special-case the sport,
because nothing in `trees/` can see it.

### `matchmind_v1.data`

Owns source rows and chronological preparation. It is the only place that knows an ATP CSV has a
column called `w_1stIn`, that the winner is always listed first, or that files arrive one season
at a time.

`matches.py` handles everything up to the point where a row becomes usable and neutral:
`load_matches`, `clean_matches`, `invalid_rows`, `order_matches`, `assign_players`, and the
`prepare_matches` composition of the last three. `dataset.py` handles the replay: `build_dataset`
produces the feature rows, `replay_state` produces just the end state, `latest_snapshots`
produces each player's last observed attributes, and `check_dataset` asserts the guarantees the
output is supposed to have.

This subsystem is pandas at its edges: `load_matches`, `clean_matches`, `order_matches`,
`assign_players` and `prepare_matches` all take and return DataFrames, and `build_dataset` returns
one too — the canonical feature set is a DataFrame written straight to CSV.

The conversion happens *inside* the replay. As `build_dataset` walks the rows, each one is turned
into `PlayerSnapshot`, `MatchContext`, `PlayerMatchStats` and `CompletedMatch` objects before any
tennis-domain logic touches it. So the domain code never receives a DataFrame row even though the
subsystem around it is frame-based, and that conversion is where values which survived the
frame-level cleaning get rejected.

### `matchmind_v1.tennis`

Owns evolving player state and pre-match feature semantics. `TennisState` holds Elo overall and
per surface, Elo histories, recent results, per-statistic serve-rate histories, match counts and
head-to-head records. `features.py` owns what a feature *means* — that `ATP_RANK_DIFF` is player
1 minus player 2, that a `LAST_k` window needs k observations on both sides — and `FEATURE_NAMES`
is the canonical order every model input follows.

This subsystem has no idea where a match came from. It takes dataclasses, not DataFrames, and it
cannot read a file.

### `matchmind_v1.trees`

Knows nothing about tennis. `decision_tree.py` imports `math`, `dataclasses` and NumPy;
`random_forest.py` imports NumPy and the tree. There is no tennis import in either file and no
scikit-learn import anywhere under `trees/`. The models see a float matrix and a binary vector,
and the fact that column 4 is a ranking difference is information they never receive.

See [Model implementation](model_implementation.md) for how they work.

### `evaluation.py`

Knows about feature matrices and calendar periods, and nothing about CSV parsing. It imports
exactly two names from the rest of the package — `TARGET_COLUMN` and `FEATURE_NAMES` — and uses
scikit-learn only for its four metric functions.

`features_and_target` is the choke point that keeps metadata out of the models: it selects
`list(FEATURE_NAMES)` by name rather than dropping known-bad columns, so a new metadata column
cannot become a model input by being forgotten about.

### `artifacts.py`

Owns serialization and, more importantly, deserialization. `save_bundle` writes deterministic
gzipped JSON; `load_bundle` treats the file as untrusted input and rebuilds a `PredictionBundle`
only if every part of it type-checks. See [Prediction bundle format](artifact_format.md).

### `prediction.py`

Consumes a frozen model plus frozen state. `PredictionBundle` holds the fitted `RandomForest`,
the `TennisState` as it stood after the last canonical match, and one `PlayerProfile` per player.
Answering a query builds one feature vector from that saved state — it does not reopen the raw
files or replay any history. `resolve` turns a name or numeric id into a profile; `predict`
scores a matchup.

### `scripts/`

Four entry points, each owning one execution path and no library logic worth reusing. They parse
arguments, print progress, write output files, and enforce the operational checks that belong to
a run rather than to a function — most notably the dataset SHA-256 gate in
`build_prediction_bundle.py`.

---

## The state / feature boundary

This is the invariant the whole pipeline is built around.

`build_features(player1, player2, context, state)` **reads** state. It calls `elo_of`,
`h2h_wins`, `serve_history_of` and their neighbours, all of which are a `dict.get` with a
default, so asking about a player who has never appeared returns a neutral value and inserts
nothing. Nothing in `features.py` writes to a `TennisState`.

`TennisState.update(match)` **advances** state. It is the only public method that mutates it —
`_append` is the private helper it calls — and it takes a `CompletedMatch`, a match whose result
is already known.

`build_dataset` therefore has exactly one correct order, and it is the order the loop uses:

```python
features = build_features(_snapshot(match, "player1"), _snapshot(match, "player2"), context, state)
rows.append(...)                                    # the row is finished here
state.update(_completed_match(match, context.surface))
```

Reversed, every row would describe a history that already includes the match being predicted. The
Elo difference would be the post-match Elo difference, the head-to-head would already count this
meeting, and the serve averages would already include this performance — which is to say the
target would be partly readable from the features, and every score downstream would be fiction.

Two tests pin this down: `test_reading_features_does_not_touch_state` compares the full state
before and after a `build_features` call, and `test_a_match_cannot_see_its_own_serve_statistics`
replays a series of matches in which one player aces 90 of every 100 service points, then checks
that the serve features stay neutral until a full window of *prior* matches exists — and that the
row where they switch on carries the value implied by the earlier matches only.

The same ordering discipline is what makes `replay_state` safe to use for deployment. It applies
the identical `update` sequence over the identical row order, so the state it ends on is exactly
the state `build_dataset` would have held after its last row.

---

## Four execution paths

### A. Dataset build — `scripts/build_dataset.py`

Reads every yearly file in `[FIRST_YEAR, LAST_YEAR]`, cleans, orders, assigns player slots,
replays state to build the feature rows, validates the result with `check_dataset`, and writes
`data/processed/match_features.csv`. Along the way it prints per-year retention, the
missing-field breakdown by group, and the before/after chronology counts, so a rebuild that
silently loses rows is visible rather than inferred.

Details in [Data pipeline](data_pipeline.md).

### B. Evaluation — `scripts/evaluate_models.py`

Reads the canonical CSV and cuts it into three fixed calendar periods with `split_by_period`.
Every candidate configuration is fitted on train and scored on validation; the lowest validation
log loss picks a configuration per family. Only then are the selected configurations refitted on
train + validation and scored once on the test period.

The script asserts that the three periods exactly tile the eligible window — no gap, no overlap,
no double-counted row — and fails rather than reporting numbers if they do not.

Results and reasoning in [Model evaluation](model_evaluation.md).

### C. Deployment build — `scripts/build_prediction_bundle.py`

A different job from evaluation, and deliberately a separate program. It verifies the dataset's
SHA-256 against a constant compiled into the script, fits the already-chosen configuration on
**all** canonical rows, separately replays the raw history to obtain the final `TennisState` and
each player's `latest_snapshots`, attaches names from `data/atp_players.csv`, and saves the
bundle.

It then reloads what it just wrote and checks the round trip: identical probabilities on a sample
of the training matrix, identical profiles, identical Elo and head-to-head state, and identical
features built against the restored state. A bundle that does not survive its own round trip is a
build failure.

Because it trains on everything, its accuracy is not a result and is not reported.

### D. Prediction — `scripts/predict_match.py`

`load_bundle` validates and rebuilds the artifact. `resolve` maps each requested player to a
profile. `predict` puts the two players in a fixed internal order, builds one feature vector from
the saved state, scores it once, and maps the answer back to the order the caller asked in.

No raw file is opened and no history is replayed. The cost of a query is one feature vector and
25 tree descents.

---

## Tracked and generated artifacts

| Path | Role |
|---|---|
| `data/all/*.csv` | Source input. Read-only; nothing rewrites a source file. |
| `data/atp_players.csv` | Source input, names only. |
| `data/processed/match_features.csv` | Generated and ignored. Rebuild it; do not archive it. |
| `results/*.csv` | Tracked evaluation output. |
| `models/matchmind_v1_bundle.json.gz` | Tracked deployment artifact. |

The feature set is the one generated file deliberately not kept, because it is a pure function of
tracked inputs and would otherwise be a second copy of the truth that can drift from the first.
Its SHA-256 is recorded instead — in `build_prediction_bundle.py`, in the bundle's own metadata,
and in [DATA_SOURCES.md](../DATA_SOURCES.md), which is where provenance lives.

---

## Architectural invariants

Each of these is enforced by code, by a test, or by both.

| Invariant | Where it holds |
|---|---|
| Models receive exactly `FEATURE_NAMES`, in that order | `features_and_target`, `feature_vector` |
| Metadata columns never enter `X` | `features_and_target` selects by name, not by exclusion |
| State advances only after its row's features are built | the `build_dataset` loop order |
| Reading state never mutates it | every `TennisState` reader is a `get` with a default |
| The canonical row order is a total order | `order_matches` ties break on `source_year`, `source_row` |
| Evaluation periods are date cuts, never shuffles | `split_by_period` |
| The three periods tile the eligible window exactly | asserted in `evaluate_models.py` |
| Test data selects nothing | selection reads validation scores only |
| The deployment refit is a separate program | `build_prediction_bundle.py` |
| A loaded bundle must match the contract it declares | `_check_metadata`, `_forest_parameters` |
| A saved bundle must reload to identical behaviour | the round-trip check in the build script |

---

## Related documentation

- [Data pipeline](data_pipeline.md) — how source rows become canonical rows
- [Feature engineering](feature_engineering.md) — what the 67 inputs mean
- [Model implementation](model_implementation.md) — how the tree and forest work
- [Model evaluation](model_evaluation.md) — how model quality was measured
- [Prediction bundle format](artifact_format.md) — how the deployable system is serialized
- [Data sources](../DATA_SOURCES.md) — where the third-party data came from
