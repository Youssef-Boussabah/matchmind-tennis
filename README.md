# MatchMind Tennis V1

Decision trees and random forests implemented from scratch, and a historical ATP match pipeline
spanning 1991 through a partial 2026 snapshot: it turns match results into pre-match features,
trains those models on them, and predicts an arbitrary matchup.

The models in `src/matchmind_v1/trees/` are written directly against NumPy — no scikit-learn or
other ML library is used in the implementation. scikit-learn is used for evaluation metrics and
to fit reference models alongside them, never inside them.

```bash
uv sync --all-groups --frozen
uv run python scripts/build_dataset.py            # data/all/*.csv  -> the feature set
uv run python scripts/evaluate_models.py          # chronological train / validate / test
uv run python scripts/build_prediction_bundle.py  # the shipped model artifact
uv run python scripts/predict_match.py --player-a "Jannik Sinner" --player-b "Carlos Alcaraz"
```

## Predicting a match

```bash
uv run python scripts/predict_match.py \
    --player-a "Jannik Sinner" \
    --player-b "Carlos Alcaraz" \
    --surface Hard \
    --best-of 3 \
    --draw-size 128
```

```
Jannik Sinner vs Carlos Alcaraz
Surface: Hard
Best of: 3

Jannik Sinner:   69.7%
Carlos Alcaraz:  30.3%

Predicted winner: Jannik Sinner

Historical data through: 2026-05-17
Profiles last observed:
  Jannik Sinner: 2026-05-06
  Carlos Alcaraz: 2026-04-13

Probabilities are uncalibrated model outputs.
```

Either player can be given as a full name or a numeric ATP id, so all 1,968 players in the
history are reachable whether or not the reference file has a name for them. Every one of them
resolves to a name in the shipped bundle. Names match exactly, ignoring case and extra spaces;
there is no fuzzy matching, and an ambiguous name is an error listing the candidate ids rather
than a guess.

Both query orders use the same underlying forest evaluation. The two players are put in a fixed
internal order — lower ATP id first — so one feature vector is scored once and that single result
is mapped back to whichever order was asked for; the two probabilities in each answer are
complements.

The same thing from Python:

```python
from matchmind_v1.artifacts import load_bundle

bundle = load_bundle("models/matchmind_v1_bundle.json.gz")
prediction = bundle.predict("Jannik Sinner", "Carlos Alcaraz", surface="Hard", best_of=3)
prediction.probability_a, prediction.winner.label
```

**Two caveats worth reading before trusting a number.** The probabilities are **uncalibrated** —
treat them as scores that rank matchups sensibly, not as literal win chances. And each player is
described by their **last observed profile**: rank, points, age and height as of their most
recent match up to 2026-05-17, which for a player who stopped earlier is that earlier date. The
CLI prints both dates.

## Models

```python
from matchmind_v1.trees import DecisionTree, RandomForest

forest = RandomForest(
    n_estimators=25,
    max_depth=8,
    min_samples_split=20,
    max_features="sqrt",
    bootstrap=True,
    random_state=0,
).fit(X, y)                    # the configuration the shipped model uses

forest.predict(X_new)          # (n_samples,) class labels
forest.predict_proba(X_new)    # (n_samples, 2) class probabilities
forest.score(X_test, y_test)   # accuracy
```

`DecisionTree` builds a CART tree using Gini impurity, searching midpoints between adjacent
unique feature values and sending `feature <= threshold` to the left child. It finds those
splits by sorting each candidate column once and reading the class counts on either side of
every threshold off a cumulative sum, rather than rebuilding a mask per threshold, which is what
makes fitting on 90,000 rows take seconds rather than hours. `RandomForest` bags those trees
over bootstrap samples with per-split feature subsampling and averages their probabilities.

Both are covered by an automated test suite, including a slow reference implementation of the
same split rule that the fast one must agree with exactly, and comparisons against scikit-learn
on synthetic problems.

## Tennis state and features

`matchmind_v1.tennis` tracks what is known about players from the matches applied so far — Elo
overall and per surface, recent form, head-to-head records, matches played, and rolling serve
statistics — and turns that history into pre-match features.

```python
from matchmind_v1.tennis import MatchContext, TennisState, build_features

state = TennisState()
features = build_features(player1, player2, MatchContext("Hard", 5, 128), state)
state.update(completed_match)
```

Feature extraction and state update are deliberately separate. `build_features` reads state and
never writes to it, so a match cannot influence its own features, and asking about a player who
has never appeared returns neutral values without inventing a history for them. `state.update`
is the only thing that advances history.

`FEATURE_NAMES` lists all 67 features in a fixed order, and `feature_vector` turns a feature
dict into an array in that order, so model inputs never depend on dictionary ordering.

## Building the dataset

`scripts/build_dataset.py` reads `data/all/atp_matches_1991.csv` … `atp_matches_2026.csv` and
writes `data/processed/match_features.csv`: eight metadata columns identifying the match, the 67
model features, and the binary `RESULT`.

The source history runs from 1991 through a partial 2026. 2025 is a complete season; the 2026
files stop at the start of Roland Garros, and since that draw carries no serve statistics the
last match the pipeline keeps is 2026-05-17. Where the files came from is recorded in
[data/SOURCE_SNAPSHOT.txt](data/SOURCE_SNAPSHOT.txt).

Three things about how it gets there:

- **Everything is globally ordered before any state is built.** The source files are not
  chronological. Some matches, Davis Cup ties among them, are listed after matches with later
  tournament dates, so reading the files in order walks backwards in time in 418 places. The
  pipeline sorts the full history by `tourney_date`, then `tourney_id`, then `match_num` before
  replaying anything. `tourney_date` is the tournament start date rather than a per-match
  timestamp. Within an event the replay therefore uses `match_num` plus the source tie-breakers;
  `round` is carried as metadata but is not part of the sort.
- **Player 1 and player 2 are assigned deterministically.** The raw rows always list the winner
  first, so the sides are swapped on a coin flip from a seeded generator, drawn in sorted-row
  order. The same inputs and seed reproduce the same slot decisions and the same rows, and the
  recorded canonical hash is reproduced in the project's pinned environment.
- **Features are built before the match updates state.** Each row is read against the history of
  every match before it, and only then applied, so nothing a match reveals about itself reaches
  its own features.

Rows missing anything the feature engine needs — a rank, a height, the serve counts — are
dropped rather than imputed, along with the handful whose recorded numbers are not physically
possible: more second serves won than second serves played, more aces than service points won, a
height of 15 cm. That keeps 99,158 of 112,768 matches. The build prints the year-by-year
retention.

## Results

Models are fitted on 1991-2021, configurations are chosen on 2022-2023, and the chosen
configurations are refitted on everything up to the end of 2023 and scored once on **2024-2025**
— 5,589 matches the selection never saw. The partial 2026 season is excluded from all three:
it is only recorded as far as May, so it is a clay-court stretch rather than a year.

| Model | Accuracy | Log loss | Brier | ROC AUC |
|---|---|---|---|---|
| Majority baseline | 0.500 | — | — | — |
| Better ATP rank wins | 0.641 | — | — | — |
| Decision tree (from scratch) | 0.637 | 0.627 | 0.219 | 0.701 |
| Random forest (from scratch) | **0.647** | **0.618** | **0.215** | **0.712** |
| Decision tree (scikit-learn) | 0.637 | 0.626 | 0.219 | 0.702 |
| Random forest (scikit-learn) | 0.649 | 0.617 | 0.215 | 0.713 |

The two baselines set the scale. Predicting the more common side is a coin flip; simply backing
the better-ranked player already gets most of the way, and the forest adds about 0.59 points on
top of that — 33 matches out of 5,589. That is a small observed margin, and since both rules
score the same matches, no significance claim is made for it either way. The scikit-learn models
are a reference implementation rather than a target — the point is that the from-scratch versions
land in the same place on the same data.

**The shipped model is not the model in that table.** The table above is the *evaluation* model:
configuration chosen on 2022-2023, refitted through the end of 2023, and scored once on
2024-2025 — seasons it had never seen.

`models/matchmind_v1_bundle.json.gz` is the *deployment* model. It takes that same frozen
configuration — the from-scratch forest, 25 trees, depth 8 — and refits it on **all 99,158
canonical matches**, the excluded 2026 rows included, with player state replayed through
2026-05-17. Freshness is what a deployment artifact is for: a question about next week's match
should be answered from the most recent history available.

That refit is **not an evaluation result**, and the 2024-2025 numbers above do not belong to it.
It has seen every match it could be scored against, so its training accuracy is meaningless as a
measure of performance and is not reported anywhere.

Full numbers, including every configuration tried on the validation period, are in
[docs/model_evaluation.md](docs/model_evaluation.md), with the machine-readable versions in
`results/test_metrics.csv` and `results/validation_results.csv`.

## Layout

```
src/matchmind_v1/trees/        decision tree and random forest
src/matchmind_v1/tennis/       Elo, player state, pre-match features
src/matchmind_v1/data/         reading the ATP files and generating the feature set
src/matchmind_v1/evaluation.py chronological splits and scoring
src/matchmind_v1/prediction.py player lookup and match prediction
src/matchmind_v1/artifacts.py  saving and loading the prediction bundle
scripts/                       dataset build, evaluation, bundle build, predict
tests/                         pytest suite
data/all/                      raw yearly ATP match files
data/processed/                the generated feature set (not committed)
models/                        the prediction bundle
results/                       current evaluation metrics, as CSV
docs/                          architecture, pipeline, features, models, evaluation, format
```

## Technical documentation

`docs/` covers the design decisions behind the code, one document per concern:

- [Architecture](docs/architecture.md) — how the subsystems fit together, and the invariants that
  hold them apart
- [Data pipeline](docs/data_pipeline.md) — how 112,768 raw rows become 99,158 ordered ones, and
  what gets rejected on the way
- [Feature engineering](docs/feature_engineering.md) — the 67 pre-match features, with the exact
  formulas and window rules
- [Model implementation](docs/model_implementation.md) — the from-scratch tree and forest,
  including the optimized split search
- [Model evaluation](docs/model_evaluation.md) — how the models were configured, scored and
  compared, with the full numbers
- [Prediction bundle format](docs/artifact_format.md) — how the shipped model and its history are
  serialized, validated and queried
- [Data sources](DATA_SOURCES.md) — provenance, licence and snapshot of the third-party data

## Development

```bash
uv sync --all-groups --frozen
uv run pytest
uv run ruff check src tests scripts
```

## Data and attribution

Historical match and player data: Jeff Sackmann / Tennis Abstract. The files live in `data/`
and are not re-fetched at build time; see [DATA_SOURCES.md](DATA_SOURCES.md) for the inventory
and provenance, including the current state of the original source repository. That data is
distributed under its own upstream licence — see [data/LICENSE.md](data/LICENSE.md).

The Elo approach follows the Tennis Abstract write-up:

- <https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/>

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for contributions.

## License

The MatchMind Tennis V1 source code is available under the [MIT License](LICENSE).

The tennis data under `data/` is third-party material and is not covered by that licence. Its
attribution and CC BY-NC-SA 4.0 terms are documented in [data/LICENSE.md](data/LICENSE.md) and
[DATA_SOURCES.md](DATA_SOURCES.md).

## A note on use

This is a portfolio and learning project. It is not built for betting, and should not be used
for it.
