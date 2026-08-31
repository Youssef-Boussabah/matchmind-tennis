# Data Pipeline

How 112,768 raw ATP match rows become 99,158 chronological machine-learning rows.

This document is about what the code does to the data. Where the data came from, which mirror and
commit it was pinned to, and what licence it carries are in [DATA_SOURCES.md](../DATA_SOURCES.md).

The pipeline has five stages, applied in this order:

```
load_matches -> clean_matches -> order_matches -> assign_players -> build_dataset
```

`prepare_matches` is the composition of the middle three, and it exists so that the deployment
build and the dataset build cannot drift apart: both call it, so both see the same rows in the
same order with the same player slots.

---

## 1. Input contract

`load_matches(input_dir, start_year, end_year)` reads `atp_matches_YYYY.csv` for every year from
`FIRST_YEAR` (1991) to `LAST_YEAR` (2026), oldest first, and concatenates them.

Three details of that function matter more than they look:

- **Every year in the range must exist.** A missing file raises `FileNotFoundError` rather than
  warning. A silently shortened history would not fail anywhere downstream; it would just produce
  a model trained on less tennis, and nothing would say so.
- **Columns are pinned at read time.** `usecols=list(REQUIRED_COLUMNS)` means a file missing any
  required column fails while it is being read, not later when a feature comes out empty.
- **Each row is tagged with its origin.** `source_year` and `source_row` are attached during the
  read. They are never model inputs; they exist to break ordering ties and to trace a canonical
  row back to the source line it came from.

`REQUIRED_COLUMNS` is the union of three groups:

| Group | Columns | Role |
|---|---|---|
| `MATCH_COLUMNS` | `tourney_id`, `tourney_name`, `surface`, `draw_size`, `tourney_date`, `match_num`, `round`, `best_of` | `best_of` and `draw_size` are model inputs; the rest identify and order the match |
| `STATIC_COLUMNS` | `winner_`/`loser_` × id, ht, age, rank, rank_points | per-player attributes as recorded at the match |
| `SERVE_COLUMNS` | `w_`/`l_` × ace, df, svpt, 1stIn, 1stWon, 2ndWon, bpSaved, bpFaced | raw serve counts |

`load_player_names` reads `data/atp_players.csv` and takes **names only**. Height, age, rank and
ranking points all come from the match rows, where they are recorded as of the match rather than
as of today — using a reference file for them would give a 1995 player their retirement-era
attributes.

---

## 2. Why the history starts in 1991

Almost every feature in this model is a serve statistic or is derived from one, and 1991 is the
first season the source records serve counts at all. Earlier rows are not merely sparse; they are
missing `svpt`, `1stIn`, `bpFaced` and the rest outright, so every one of them would be dropped
by the cleaner anyway. Starting the range at 1991 states that boundary instead of discovering it.

Provenance and coverage are in [DATA_SOURCES.md](../DATA_SOURCES.md).

---

## 3. Validation and rejection

Validation happens in two stages, and the difference between them is deliberate.

### Stage one — frame-level filtering in `clean_matches`

Rows that fail here are **dropped**. In order:

1. **Numeric coercion.** Every integer and float column goes through `pd.to_numeric(...,
   errors="coerce")`, so a non-numeric value becomes `NaN` rather than a string that later
   compares strangely.
2. **Calendar dates.** `_is_calendar_date` keeps only rows whose `tourney_date` parses as a real
   eight-digit `YYYYMMDD` day. This also removes rows with no date at all.
3. **Required-field completeness.** `dropna` over the full `REQUIRED_COLUMNS` list. A row missing
   any one of them is gone.
4. **Whole-valued integers.** `_whole_numbers` requires every integer column to hold a value with
   no fractional part *before* the cast to `int`. A serve count of `100.9` is not a count written
   as a float; it is a corrupt value, and truncating it to `100` would repair data the pipeline
   exists to exclude.
5. **Physical impossibility**, via `invalid_rows`.

`invalid_rows` is where the tennis-specific arithmetic lives. For each side independently:

| Check | Why it must hold |
|---|---|
| no negative serve count | counts |
| `ace <= svpt` | you cannot ace more points than you served |
| `df <= svpt` | same |
| `1stIn <= svpt` | first serves in are a subset of serves |
| `1stWon <= 1stIn` | you can only win a first-serve point you landed |
| `2ndWon <= svpt - 1stIn` | second serves played is what is left over |
| `2ndWon + df <= svpt - 1stIn` | a double fault consumes a second serve, and it is not a win |
| `ace <= 1stWon + 2ndWon` | an ace is a service point won |
| `bpSaved <= bpFaced` | you cannot save one you never faced |
| `bpFaced <= svpt` | break points are service points |

Plus two row-level checks: the same player cannot appear on both sides, and each height must fall
inside `[MIN_HEIGHT_CM, MAX_HEIGHT_CM]` = `[100, 250]`. Those bounds are wide on purpose — they
catch a value recorded in the wrong unit or not at all, not an unusually tall player.

### Stage two — object-level assertion during `build_dataset`

`PlayerSnapshot`, `MatchContext` and `PlayerMatchStats` re-check their own inputs in
`__post_init__`, and they **raise** rather than filter:

- `PlayerSnapshot`: `player_id` a positive integer, `age > 0`, `atp_rank >= 1`, `atp_points >= 0`,
  height inside the same physical range.
- `MatchContext`: `surface` one of `SURFACES`, `best_of` exactly 3 or 5, `draw_size` an integer of
  at least 2. Draw sizes are not all powers of two — the source contains 9, 10, 12, 18, 24 and 28.
- `PlayerMatchStats`: the same arithmetic as `invalid_rows`, expressed per object.

The overlap is intentional. Stage one is a filter over a frame that may contain anything; stage
two is a guarantee at the point of use, and it holds for callers who never went through
`clean_matches` at all — the prediction path builds a `PlayerSnapshot` from a saved profile, and
`MatchContext` from whatever the CLI was given. Since the real build completes, every one of the
99,158 canonical rows satisfies both stages.

### Why rejection rather than imputation

Every historical feature in this project is an *observation* about a player: what fraction of
their break points they saved over their last 50 recorded observations
(`P_BP_SAVED_LAST_50_DIFF`), what their first-serve-won rate has been over the last 25
(`P_1ST_WON_LAST_25_DIFF`), what their Elo did over their last ten (`ELO_GRAD_LAST_10_DIFF`).

Filling in a missing `bpFaced` with a column mean does not produce a slightly noisier row. It
produces a break-point record that was never played, and `state.update` then appends that
fabricated rate to the player's `break_points_saved` deque, where it sits inside every
`P_BP_SAVED_LAST_k_DIFF` window that player appears in until as many as 200 subsequent *recorded*
observations of that same statistic displace it. Because a match in which the player faced no
break point appends nothing to that deque, those 200 observations can span considerably more than
200 matches — the six serve deques are statistic-specific and fill at different rates. The state
replay carries an imputed value forward in a way a row-independent model would not.

Dropping is not free either, and it is worth being straight about what it costs. `state.update`
runs only for retained rows, so a dropped match does not just vanish from the supervised CSV — it
never advances overall Elo, surface Elo, recent results, matches played, head-to-head, surface
head-to-head or the serve histories. Two players who met in a dropped match have no record of it,
and their ratings never move for it.

So the real trade is: dropping sacrifices the match *and its downstream contribution to history*,
while imputing invents statistics and lets fabricated values propagate into that same history. The
project takes the first side of it every time. That is a conservative contract, chosen because a
gap is easier to reason about than a fabrication — not a claim that dropping is statistically
optimal.

`retention_by_year` and `missing_by_group` report where the losses go — by season, and by
overlapping field group — so the shape of what was dropped is visible in every build. 13,610 rows
of 112,768 do not survive, most of them serve statistics missing in the early 1990s; the coverage
is not uniform across the period, which is recorded as a limitation in
[model_evaluation.md](model_evaluation.md).

---

## 4. Chronological ordering

`order_matches` performs a single stable sort on `ORDER_COLUMNS`:

```python
ORDER_COLUMNS = ("tourney_date", "tourney_id", "match_num") + ("source_year", "source_row")
```

In order of application:

| Key | What it contributes |
|---|---|
| `tourney_date` | the real chronology — the day the tournament started |
| `tourney_id` | groups a tournament's matches together within a shared start date |
| `match_num` | the within-tournament order — the only such key this sort uses |
| `source_year` | tie-break: which yearly file the row came from |
| `source_row` | tie-break: the row's position in that file |

**The source files are not chronological, and not only at the seams.** The files are grouped by
season, so concatenating them oldest-to-newest does put the seasons in the right order — but an
individual season file is not itself in date order, and that is where the whole problem lives.

Walking the cleaned rows in the order the files supply them steps backwards in time in 418 places.
Every one of those 418 steps falls **inside a single year file**; not one occurs at a boundary
between two files:

| Backward date steps in the cleaned concatenated order | Count |
|---|---:|
| Within one season file | 418 |
| Across a season boundary | 0 |

The seasons that contribute them, measured on the cleaned rows of each file:

| Year | Steps | Year | Steps | Year | Steps |
|---|---:|---|---:|---|---:|
| 2000 | 29 | 2007 | 33 | 2019 | 12 |
| 2001 | 32 | 2008 | 31 | 2020 | 1 |
| 2002 | 31 | 2009 | 31 | 2021 | 46 |
| 2003 | 29 | 2017 | 1 | 2022 | 1 |
| 2004 | 31 | | | 2023 | 17 |
| 2005 | 32 | | | 2024 | 27 |
| 2006 | 33 | | | 2026 | 1 |

Total 418. The jumps are not small: in the 2021 file, Paris Masters (`20211101`) is immediately
followed by Montpellier (`20210222`), and Indian Wells (`20211004`) by Rotterdam (`20210301`) —
backwards by most of a season.

Every row of a `tourney_id` carries the same `tourney_date`, so a *backward date step* can only
occur between events, never inside one. That is a statement about the date key alone, not about
what the sort does to rows: within an event the sort still assigns a definite order, from
`match_num` and then the two source tie-breakers. Ordering inside a tournament is therefore
deterministic — it is simply not derived from a played-at timestamp, which the source does not
carry.

So sorting is not a tidy-up applied to nearly-ordered input. Concatenating the seasons in year
order is genuinely insufficient, and `order_matches` is what establishes the global chronology the
state replay requires. After it, the count is 0 and no row has a later-dated match above it.

Replaying unsorted history is not a cosmetic problem. `state.update` is applied in whatever order
the loop sees, so a match listed early but played late writes its Elo, its head-to-head and its
serve rates into the state *before* matches that were actually played earlier.

If an out-of-order future match advances state for a player who appears in a chronologically
earlier row, that earlier row can read information that should not yet exist. Not every earlier row
is affected — only those whose features intersect the state the future match advanced, normally
because one of its two players was involved. But the contamination is silent: the resulting
evaluation is temporally contaminated, and downstream scoring cannot recover the correct chronology
from features that already encode it. How much any particular metric moves, and in which direction,
is not something this ordering argument establishes.

**What the sort guarantees.** The order is total and reproducible: the last two keys are unique
per row, so no two rows can tie, and re-sorting the same rows gives the same sequence. It does not
depend on how pandas happened to concatenate the yearly frames.
`test_ordering_the_same_rows_twice_gives_the_same_order` covers this.

**What it does not guarantee.** `tourney_date` is a tournament start date, not a per-match
timestamp. Every match in a two-week event shares one date, so within a tournament the order rests
entirely on `match_num`, which is a draw position rather than a clock. Two matches played on
different days of the same event may be replayed in the wrong relative order.

The source does carry one other within-event signal: `round` is kept as a metadata column and its
values (`R128`, `R64`, `R32`, `R16`, `QF`, `SF`, `F`, plus `RR` and `BR`) encode a coarse
progression through the draw. `order_matches` does not sort on it — round-robin and bronze-match
labels do not slot cleanly into a single progression, and `match_num` is finer-grained where the
two agree — so it is available in the dataset but takes no part in the ordering.

Period boundaries are unaffected by any of this, since they are cut on the date itself, but the
within-tournament state is approximate. This is recorded as a limitation in
[model_evaluation.md](model_evaluation.md#limitations).

---

## 5. Player slot neutralization

Raw ATP rows always list the winner first. Handed to a model unchanged, `winner_rank` and
`loser_rank` would make the target trivially readable from the column layout: the model would
learn the file format, score near 100%, and know nothing about tennis.

`assign_players` breaks that:

```python
rng = np.random.default_rng(seed)          # seed defaults to 0
player1_won = rng.random(len(matches)) < 0.5
...
assigned[f"player1_{name}"] = np.where(player1_won, winner, loser)
assigned[f"player2_{name}"] = np.where(player1_won, loser, winner)
assigned[TARGET_COLUMN] = player1_won.astype(int)
```

One draw per row decides which side that match's winner lands on, and **every** paired field moves
together — id, height, age, rank, ranking points and all eight serve counts. `RESULT` is defined
as exactly that flip: `1` when the winner was placed in slot 1, `0` when they were placed in slot
2. There is no separate notion of who won; the target *is* the slot assignment.

What is deterministic about it: the draws come from `np.random.default_rng(seed)` and are consumed
in row order, so the same rows in the same order with the same seed produce the same slot
decisions and the same assigned values. `assign_players` returns a DataFrame, so what is
deterministic here is the `player1_won` sequence and the values derived from it, not a byte
stream. That is a reproducibility property, not a security one — this is a seeded pseudo-random
generator, and nothing here is cryptographic.

What it does not claim: individual rows are not balanced or stratified. The flips are independent
fair coins, so the overall split lands near 50/50 by law of large numbers rather than by
construction — the canonical dataset is 50.07% player-1 wins on the training period, which is why
the majority baseline sits at chance.

Row order is untouched by this step. `assign_players` copies the match and source columns across
unchanged and only rewrites which side each player's values sit on;
`test_slot_assignment_leaves_the_row_order_alone` holds it to that.

---

## 6. Building one canonical row

For a single match, `build_dataset` does this:

```
raw row (winner/loser already neutralised into player1/player2)
   |
   +-- MatchContext(surface, best_of, draw_size)
   +-- PlayerSnapshot(player1_id, atp_points, atp_rank, age, height)
   +-- PlayerSnapshot(player2_id, ...)
   |
   v
build_features(p1, p2, context, state)      <- reads state as it stands
   |
   v
[8 metadata] + [67 features in FEATURE_NAMES order] + [RESULT]
   |
   v
state.update(CompletedMatch(..., player1_stats, player2_stats))   <- only now
```

The resulting schema is `DATASET_COLUMNS`, exactly 76 columns:

| Block | Count | Contents |
|---|---|---|
| `METADATA_COLUMNS` | 8 | `tourney_date`, `tourney_id`, `match_num`, `tourney_name`, `surface`, `round`, `player1_id`, `player2_id` |
| `FEATURE_NAMES` | 67 | the model inputs, in fixed order |
| `TARGET_COLUMN` | 1 | `RESULT` |

The metadata block is not evidence about who wins; it identifies a match so a row can be traced,
partitioned by date, and checked for chronology. Keeping it in a separate tuple from
`FEATURE_NAMES` is what lets `features_and_target` select model inputs by name and be certain
nothing else came along.

Feature values are written into the row with `[features[name] for name in FEATURE_NAMES]` — the
same order contract `feature_vector` uses at prediction time. Dictionary iteration order is never
the model contract. See [Feature engineering](feature_engineering.md).

`check_dataset` then asserts the output's guarantees: the exact column list, no duplicate columns,
all 67 feature values finite, `RESULT` binary, no match with the same player on both sides, and
`tourney_date` monotonically increasing.

---

## 7. The source bound and the canonical bound

Two dates are easy to confuse, and they differ:

| | Date |
|---|---|
| Latest `tourney_date` in the raw source files | 20260525 |
| Latest `tourney_date` in the canonical dataset and the deployment bundle | 20260517 |

The 2026 files stop during Roland Garros, and the 127 Roland Garros 2026 rows carry no serve
statistics. They fail the required-field check in `clean_matches` and are dropped, so the last
match the pipeline keeps is a week earlier.

Nothing special-cases this. It is the ordinary rejection rule producing a visible effect at the
edge of the data, and it is worth stating because the two dates appear in different files: 20260525
in `data/SOURCE_SNAPSHOT.txt`, 20260517 in the bundle's `history_end_date` and in the prediction
CLI's output.

---

## 8. Determinism and reproducibility

The row values and their ordering are fully determined by the inputs and the seed, and six
things make that true:

1. **Fixed input files.** Nothing is downloaded at build time; the pipeline reads only what is in
   `data/`, and it never rewrites a source file.
2. **An explicit source range.** `FIRST_YEAR` to `LAST_YEAR`, with a missing year an error rather
   than a shorter history.
3. **A total sort order.** `ORDER_COLUMNS` ends with two per-row-unique keys, so no tie can be
   settled by chance.
4. **A fixed slot-assignment seed.** Seed 0 by default, drawn in sorted-row order.
5. **A fixed feature order.** `FEATURE_NAMES` is a module-level tuple, not a dict view.
6. **Deterministic state updates.** `TennisState.update` mutates the state, but it does so by
   deterministic arithmetic over the row sequence — no clock, no hashing of unordered containers,
   no floating-point reduction over an unspecified order. The same ordered rows produce the same
   state transitions.

Those six fix the *content* of the dataset. The bytes of the CSV additionally depend on pandas'
serialization, so the recorded hash below is reproduced in the project's pinned environment rather
than being claimed for arbitrary pandas and Python versions.

The current canonical dataset is 99,158 rows × 76 columns, SHA-256
`22425a372212c475ff0eacaf7a85916cc2468d16a504641ad06883754ddbe1ef`. That hash is compiled into
`scripts/build_prediction_bundle.py` as `EXPECTED_DATASET_SHA256`, and the deployment build refuses
to run against anything else — a stale or hand-edited feature file cannot quietly become the
shipped model.

---

## Related documentation

- [Feature engineering](feature_engineering.md) — what `build_features` produces from the replay
- [Architecture](architecture.md) — how this stage connects to evaluation and deployment
- [Model evaluation](model_evaluation.md) — how the canonical rows are partitioned and scored
- [Data sources](../DATA_SOURCES.md) — provenance, licence and snapshot of the raw files
