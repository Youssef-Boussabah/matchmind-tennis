# Prediction Bundle Format

How a trained forest and thirty-five years of replayed tennis history become one portable file,
and what has to be true of that file before the package will predict with it.

Everything here comes from `src/matchmind_v1/artifacts.py` and `prediction.py`.

---

## 1. Why JSON and gzip

The bundle is a single JSON document, gzip-compressed. Not pickle, not joblib, not a NumPy
archive.

The concrete things this implementation gets from that choice:

- **An explicit schema.** Every field is written by a named function and read by a named function.
  There is no "whatever attributes the object happened to have at save time" — adding a field to
  `RandomForest` does not silently change the artifact, and removing one does not silently break
  it at load.
- **Inspectable primitives.** The payload is numbers, strings, lists and objects. A bundle can be
  decompressed and read with any JSON tool, on any machine, without importing this package or
  matching its version.
- **Strict validation is possible.** Because loading is a parse rather than an object
  reconstruction, every value passes through a checker that can reject it. Section 9 covers what
  that buys.
- **Stable serialization.** Sorted keys and a fixed separator make the emitted JSON text a function
  of the payload's content alone. Section 8.
- **No Python object-code deserialization.** Restoring the bundle follows explicit reconstruction
  code for the package's own model and state classes — `PredictionBundle`, `RandomForest`,
  `DecisionTree`, `_Node`, `TennisState`, `PlayerProfile`, `PlayerSnapshot` — plus primitive
  containers such as `deque`, `dict` and `list`. The payload does not name arbitrary Python classes
  or constructors to execute; it supplies values that this module's own code decides what to do
  with. Loading a pickle, by contrast, executes instructions from the file to decide what to
  construct. That is the specific property being claimed here — not that JSON is inherently safe.

The cost is size: the artifact is 3,997,928 bytes, and the state histories are most of it. That is
a reasonable price for a file that can be diffed, inspected and validated.

---

## 2. What the bundle contains

Four top-level sections:

```json
{
  "metadata": { "bundle_version": 1, "feature_names": ["BEST_OF", ...], ... },
  "model": {
    "n_features": 67,
    "parameters": { "n_estimators": 25, "max_depth": 8, ... },
    "trees": [ { "n_features": 67, "root": { ... } }, ... ]
  },
  "state": {
    "elo":            [[<player_id>, <rating>], ...],
    "surface_elo":    { "Hard": [[<player_id>, <rating>], ...], "Clay": [...] },
    "matches_played": [[<player_id>, <count>], ...],
    "elo_history":    [[<player_id>, [<rating>, ...]], ...],
    "recent_results": [[<player_id>, [0, 1, ...]], ...],
    "serve_history":  [[<player_id>, { "ace": [<rate>, ...], "double_fault": [...] }], ...],
    "h2h":            [[<winner_id>, <loser_id>, <wins>], ...],
    "surface_h2h":    { "Hard": [[<winner_id>, <loser_id>, <wins>], ...], ... }
  },
  "profiles": [ { "player_id": ..., "age": ..., "name": ... }, ... ]
}
```

(Schematic — angle brackets stand in for values, and the real file is one line with no
whitespace.)

Player-keyed dictionaries become **sorted lists of pairs**, because JSON object keys must be
strings and round-tripping integer ids through strings invites exactly the kind of type drift the
loader spends its time preventing. Head-to-head records become triples `[winner, loser, wins]` for
the same reason — the key is a pair of ids.

The current bundle holds 25 trees, 67 features, 1,968 profiles, and state replayed to
`tourney_date` 20260517.

---

## 3. Metadata contract

Written by `build_prediction_bundle.py` and checked by `_check_metadata`:

| Field | Type | Meaning |
|---|---|---|
| `bundle_version` | int | format version; must equal `BUNDLE_VERSION` |
| `package_version` | non-empty string | `matchmind_v1.__version__` at build time |
| `dataset_sha256` | non-empty string | the canonical feature set this model was trained on |
| `dataset_rows` | int ≥ 1 | rows in that dataset |
| `feature_names` | list | must equal `FEATURE_NAMES` exactly |
| `model_parameters` | object | a second copy of the forest's settings — see section 4 |
| `history_start_date` | int | first `tourney_date` in the replayed history |
| `history_end_date` | int | last `tourney_date` in the replayed history |
| `profile_count` | int ≥ 1 | must equal the number of profiles actually present |

`bundle_version` is added by `save_bundle` itself rather than by the caller, so it always describes
the writer.

**`feature_names` must equal `FEATURE_NAMES` element for element, in order** — the check is
`tuple(names) != FEATURE_NAMES`, not a set comparison. A tree model stores splits as "column 4
≤ −11.5"; that column index is meaningful only against a specific ordering. A bundle built when
the feature list was different, or ordered differently, would still load, still predict, and still
be wrong, so it is refused instead. This is the load-time half of the order contract described in
[feature engineering](feature_engineering.md#11-why-the-order-is-fixed).

Two cross-checks run outside `_check_metadata`, in `_bundle_from_payload`:

- `profile_count` must match `len(profiles)`.
- Every player with an Elo rating must have a profile. A rated player with no profile could be
  reached by the state but not resolved by a query, so it is an error naming the first offender.

The current metadata:

```
bundle_version      1
package_version     1.0.0
dataset_rows        99158
dataset_sha256      22425a...ddbe1ef
history_start_date  19910107
history_end_date    20260517
profile_count       1968
```

---

## 4. The model parameter contract

`MODEL_PARAMETERS` names the seven settings that define the forest:

```python
("n_estimators", "max_depth", "min_samples_split", "min_impurity_decrease",
 "max_features", "bootstrap", "random_state")
```

They are stored **twice** — once in `metadata.model_parameters`, once in `model.parameters` — and
`_forest_parameters` reconciles the two copies on four separate grounds:

1. **Both must be objects.**
2. **Both must be complete.** Every one of the seven must be present in each copy. A missing
   parameter would be filled in from the `RandomForest` constructor default, quietly producing a
   different model from the one that was saved — the failure would be silent and the artifact would
   still work, which is the worst combination.
3. **Neither may carry extras.** An unexpected key means the file was written by something with a
   different idea of what configures this model, so it is refused rather than ignored.
4. **The two copies must match on value *and* type.**

That last check exists because of a specific Python trap:

```python
8 == 8.0     # True
1 == True    # True
0 == False   # True
```

A bundle whose metadata says `max_depth: 8.0` while its forest says `max_depth: 8`, or whose
metadata says `bootstrap: 1` while its forest says `bootstrap: true`, would pass a value-only
comparison. But those are not the same declaration, and a file that describes its own model two
different ways is not one to predict with. The comparison is therefore:

```python
if type(kept) is not type(said) or kept != said:
```

`test_a_parameter_of_the_wrong_type_is_refused` covers all six of the confusable cases, including
`min_impurity_decrease: 0` against `0.0` — the float direction, which is just as wrong.

The stored copy is what the forest is actually rebuilt from; the metadata copy is what the bundle
*claims*. Requiring them to agree makes the human-readable metadata trustworthy rather than
decorative.

Finally, the tree count is checked against the declared one: `len(model.estimators)` must equal
`n_estimators`.

---

## 5. Tree serialization

A tree is `{"n_features": int, "root": node}`. A node is written by `_node_to_json`:

```python
saved = {"prediction": node.prediction, "probability": node.probability}
if not node.is_leaf:
    saved["feature"] = node.feature
    saved["threshold"] = node.threshold
    saved["left"] = _node_to_json(node.left)
    saved["right"] = _node_to_json(node.right)
```

Every node carries `prediction` and `probability`; a split node adds the four keys in `SPLIT_KEYS`
— `feature`, `threshold`, `left`, `right`. Leaf-ness is encoded by *absence*, matching the
in-memory representation where `feature is None` is the leaf test. There is no separate flag that
could contradict the structure.

`_node_from_json` reverses it with checks at every step. `prediction` must be an integer that is 0
or 1. `probability` must be a finite number in `[0, 1]`. Then the split keys are counted:

```python
present = [key for key in SPLIT_KEYS if key in data]
if not present:
    return node                      # a leaf
if len(present) != len(SPLIT_KEYS):
    raise ValueError(f"split node is missing {...}")
```

All four or none — a node with a `feature` but no `right` child is a corrupt node, not a leaf with
extra decoration. `feature` must then be an integer in `[0, n_features)`, so a split can never
reference a column the model does not have, and `threshold` must be a finite number. Children
recurse under the same rules.

`n_features` is verified at two levels: each tree's must equal the forest's, and the forest's must
equal `len(FEATURE_NAMES)`.

The forest itself is reconstructed by calling `RandomForest(**parameters)` and then assigning
`n_features_` and `estimators` directly. Nothing is refitted; the saved trees *are* the model.

---

## 6. State serialization

This section is the largest part of the file, and the reason it exists is worth stating plainly.

**A trained forest is not enough to predict with.** The model consumes 67 numbers, and almost all
of them are differences of *historical state*: Elo, surface Elo, matches played, head-to-head,
rolling Elo trends, rolling win counts, rolling serve rates. To build a feature vector for
"Sinner against Alcaraz on hard court tomorrow", the system needs both players' accumulated
history as of the end of the data. Ship only the forest and every query would have to reload the
raw CSVs and replay 99,158 matches to reconstruct it.

So the bundle carries the `TennisState` as it stood after the last canonical match. All eight
fields are persisted:

| Payload key | In-memory type | Serialized as |
|---|---|---|
| `elo` | `dict[int, float]` | `[[player_id, rating], ...]`, id-sorted |
| `surface_elo` | `dict[str, dict[int, float]]` | surface-keyed object of the same |
| `matches_played` | `dict[int, int]` | `[[player_id, count], ...]` |
| `elo_history` | `dict[int, deque[float]]` | `[[player_id, [rating, ...]], ...]` |
| `recent_results` | `dict[int, deque[int]]` | `[[player_id, [0 or 1, ...]], ...]` |
| `serve_history` | `dict[int, dict[str, deque[float]]]` | `[[player_id, {stat: [rate, ...]}], ...]` |
| `h2h` | `dict[tuple[int, int], int]` | `[[winner, loser, wins], ...]` |
| `surface_h2h` | `dict[str, dict[tuple, int]]` | surface-keyed object of the same |

Restoration rebuilds the deques with their original `maxlen=MAX_HISTORY` (200), so a loaded state
behaves identically as history continues — `test_histories_come_back_capped` checks the cap
survives.

The values are checked as they are read: Elo entries must be finite numbers, match counts
non-negative integers, recent results integers in `[0, 1]`, and serve rates numbers in
`[0, MAX_SERVE_RATE]` where `MAX_SERVE_RATE` is 100.0, matching the 0–100 scale
`_serve_rates` produces. Surface keys must be in `SURFACES`, serve-statistic keys must be in
`SERVE_STATS`, no player may be listed twice, and no history may be longer than `MAX_HISTORY`.

---

## 7. Player profiles

A `PlayerProfile` is a player's **last observed** attributes:

```python
{"player_id": ..., "atp_points": ..., "atp_rank": ..., "age": ...,
 "height": ..., "last_seen_date": ..., "name": ...}
```

The `PlayerSnapshot` fields — points, rank, age, height — feed the four static difference features.
`last_seen_date` records when they were observed, and `name` comes from `data/atp_players.csv` and
may be `null` for a player the reference file does not cover.

Profiles are what make a query cheap. Combined with the saved state, they mean answering "A against
B" needs no raw data at all: `resolve` looks up two profiles, `build_features` reads the saved
state, and one feature vector goes to the forest. `PredictionBundle.__post_init__` also builds a
normalised-name index so a name can be resolved without scanning, with ambiguous names raising an
error that lists the candidate ids rather than guessing.

The "last observed" part is a real caveat, not a formality. A player who stopped competing in 2015
carries their 2015 rank and age, and the prediction CLI prints each player's `last_seen_date`
alongside the history end date so the gap is visible.

All 1,968 players in the current bundle resolve to a name.

---

## 8. Deterministic bytes

`save_bundle` removes the sources of nondeterminism that are under the project's control — key
order, whitespace, the build clock and the output filename — and repeated builds in the validated
environment reproduce the same bytes. Five decisions do that work, each addressing a specific
source of variation:

```python
text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

with open(path, "wb") as raw, gzip.GzipFile(
    fileobj=raw, mode="wb", compresslevel=9, filename="", mtime=0
) as compressed:
    compressed.write(text.encode("utf-8"))
```

| Decision | What it removes |
|---|---|
| `sort_keys=True` | dictionary insertion order — keys are emitted in sorted order regardless of how the dict was built |
| `separators=(",", ":")` | whitespace variation between encoders |
| `allow_nan=False` | a `NaN` would be written as the JSON-invalid bare literal `NaN`; a model that produced one has no business being saved, so this raises instead |
| `mtime=0` | the build clock — gzip normally stamps the current time into bytes 4–8 of the header |
| `filename=""` | the source path — gzip normally stores the output filename in the header |

Passing an explicit `GzipFile` over an already-open handle, rather than `gzip.open(path, "wb")`, is
what makes the last two controllable: `gzip.open` would infer the filename from the path, so the
artifact's bytes would depend on where it was written.

Two tests hold this. `test_saving_twice_gives_the_same_bytes` compares SHA-256 of two saves, and
`test_the_gzip_header_carries_no_clock_or_filename` reads the header directly — bytes 4–8 must be
zero, and flag bit 3 must be clear. `test_the_json_holds_no_nan_or_infinity` decompresses the
payload and greps for the three non-finite literals.

**What this does and does not establish.** The JSON text is a pure function of the payload, and the
gzip header carries no clock or path, so nothing the project controls varies between builds. The
compressed bytes additionally depend on the DEFLATE encoder behind `gzip`, which the project pins
only through its declared environment — it has not been tested across other Python or zlib builds,
and no claim is made that every implementation everywhere would emit the same compressed stream.
The verified statement is the practical one: rebuilding in the project's environment reproduces the
recorded SHA-256, which was checked by building a second copy and comparing. Portability is
unaffected either way, since any correct gzip reader decompresses the file to the same JSON.

`build_prediction_bundle.py` goes one step further at build time: it reloads the file it just wrote
and verifies that the restored model gives identical probabilities on a sample of the training
matrix, that profiles and state compare equal, and that features built against the restored state
match the originals.

The current artifact:

| | |
|---|---|
| Size | 3,997,928 bytes |
| SHA-256 | `2b97e793fc946bf27b1a024ab46608264d1623d5927b6db21fd43d7ce9b22d24` |
| Trees | 25 |
| Features | 67 |
| Profiles | 1,968 |
| History end | 20260517 |

---

## 9. Loader hardening

`load_bundle` treats the file as untrusted input. It may have been hand-edited, truncated, written
by a different version, or produced by something else entirely. The governing rule is that a value
of the wrong *kind* is rejected rather than converted into the right one — no string is parsed into
a number, no float is rounded into an integer, no boolean is accepted as a count.

The one conversion that does happen is a widening: a field holding a real number accepts a JSON
integer and returns it as a Python `float`. Section "Type strictness" below states exactly where
the line falls.

Every failure reaches the caller as a `ValueError`, so `except ValueError` catches all of them —
which is what `scripts/predict_match.py` relies on. But they arrive by three different routes, and
only two of them attach the file path:

| Route | Trigger | Message |
|---|---|---|
| Wrapped as *not a readable prediction bundle* | `OSError`, `EOFError`, `UnicodeDecodeError`, `json.JSONDecodeError` — not gzip, truncated, not UTF-8, not JSON syntax | path + underlying error |
| Wrapped as *not a well-formed prediction bundle* | `KeyError`, `IndexError`, `TypeError`, `AttributeError` raised inside `_bundle_from_payload` — a missing section or a container of the wrong shape | path + underlying error |
| Not wrapped — propagates as raised | every deliberate `ValueError` from the validators | the specific validation message, **without** the path |

That third route is the common one, and it is deliberate. The two `except` clauses in `load_bundle`
list only the exception types that indicate *structural* trouble; a plain `ValueError` is in
neither tuple, so the validators' own messages travel to the caller intact rather than being
flattened into a generic "not well-formed". Observed on the shipped bundle:

```
tampered leaf probability   -> leaf probability must be between 0 and 1, got 5.0
tampered bundle_version     -> bundle version 2 is not supported, expected 1
a 1e999 literal             -> bundle contains the out-of-range number 1e999
a bare NaN literal          -> bundle contains the non-finite value NaN
not a gzip file             -> <path> is not a readable prediction bundle: Not a gzipped file
a deleted "state" section   -> <path> is not a well-formed prediction bundle: 'state'
```

Note that `_parse_float` and `_reject_constant` raise from *inside* the parsing `try` block, and
still propagate unwrapped, because a plain `ValueError` is not a `JSONDecodeError`.

The validators on that third route are `_json_int`, `_json_number`, `_json_text`,
`_check_metadata`, `_forest_parameters`, `_node_from_json`, the state and profile parsers, the
cross-checks in `_bundle_from_payload`, and `_check_predicts`.

### Strict number parsing

Python's JSON reader is more permissive than the JSON specification, in two ways that matter:

```python
payload = json.loads(text, parse_float=_parse_float, parse_constant=_reject_constant)
```

- `_reject_constant` refuses the bare literals `NaN`, `Infinity` and `-Infinity`, which Python
  accepts by default and which are not valid JSON.
- `_parse_float` refuses any float literal that parses to a non-finite value. `1e999` is
  *syntactically valid* JSON and parses to `inf`; nothing in this artifact may hold one.

Both hooks fire during parsing, so a non-finite value never reaches the reconstruction code at all.

### Type strictness

Two helpers do most of the work, and they use `type(value) is not ...` rather than `isinstance`:

```python
def _json_int(value, name, *, minimum=None, maximum=None):
    if type(value) is not int:                    # a whole-valued float is not an int
        raise ValueError(...)

def _json_number(value, name, *, minimum=None, maximum=None):
    if type(value) not in (int, float):           # "8" is not a number, however numeric it looks
        raise ValueError(...)
```

`isinstance(True, int)` is `True` in Python, so `isinstance` would let a boolean through anywhere
an integer is expected. Exact type comparison closes that.

The two helpers are deliberately not equally strict, and the difference is worth being precise
about:

| Input | `_json_int` | `_json_number` |
|---|---|---|
| `8` | `8` | `8.0` — **widened** |
| `8.0` | rejected — "must be a whole number" | `8.0` |
| `True` | rejected | rejected |
| `"8"` | rejected | rejected |

So `_json_int` is exact: an integer-only field such as `dataset_rows`, `profile_count`,
`last_seen_date`, a match count, a leaf `prediction` or a split `feature` index must arrive as a
JSON integer, and `99158.0` is refused. `_json_number` is looser by one step: a field that
conceptually holds a real number — an Elo rating, a serve rate, a profile's age, a split threshold
— accepts either JSON form and normalises to `float`, so a profile age written as `25` loads as
`25.0`. That is a widening within the numeric domain, not a coercion across kinds; strings and
booleans are still refused by both.

Exact type strictness matters most in three places: fields that are conceptually integers, the
boolean-versus-integer distinction, and the two duplicated model-parameter declarations, where
type **and** value must match across the copies — see [section 4](#4-the-model-parameter-contract).

### What is validated

| Area | Checks |
|---|---|
| Metadata | version equality, feature-list equality, required fields present and correctly typed, `history_start_date <= history_end_date` |
| Model parameters | both copies complete, no extras, values and types equal across copies |
| Forest | `n_features` equals 67, tree list non-empty, tree count equals `n_estimators` |
| Trees | per-tree `n_features` equals the forest's |
| Nodes | prediction ∈ {0, 1}; probability finite in [0, 1]; all four split keys or none; feature index inside range; finite threshold |
| State | player ids positive integers, no duplicates, match counts ≥ 0, results ∈ {0, 1}, serve rates in [0, 100], histories no longer than 200, only known surfaces, only known serve statistics |
| Profiles | list of objects, positive unique ids, all numeric fields finite, name a string or `null`, and every field re-validated by `PlayerSnapshot.__post_init__` |
| Cross-checks | `profile_count` matches, every rated player has a profile |
| Live check | `_check_predicts` |

`_check_predicts` is the last gate. It runs one prediction on a zero vector of the right width and
requires the result to be shaped `(1, 2)`, finite, non-negative and summing to 1. Everything before
it validates the file's *description* of a model; this confirms the reassembled object actually
functions.

---

## 10. Query order and complements

`PredictionBundle.predict(player_a, player_b, ...)` has a detail worth reading, because it is the
difference between a coherent answer and two unrelated ones.

The public request can arrive either way round — "Sinner against Alcaraz" or "Alcaraz against
Sinner". Internally, the two players are put in a **fixed order based on ATP id**, lowest first:

```python
a_is_first = profile_a.player_id < profile_b.player_id
first, second = (profile_a, profile_b) if a_is_first else (profile_b, profile_a)

features = build_features(first.snapshot, second.snapshot, context, self.state)
probability_first = float(self.model.predict_proba(feature_vector(features).reshape(1, -1))[0, 1])

probability_a = probability_first if a_is_first else 1.0 - probability_first
```

**One orientation is built. One feature vector is scored. Once.** The answer is then mapped back to
whichever order the caller asked in, and `probability_b` is defined as `1.0 - probability_a`, so
the two halves of a single answer sum to exactly 1.0 by construction rather than by luck.

Why this matters: swapping the players does not simply negate the feature vector, and even where
it does the forest is under no obligation to mirror its answer.

The 65 player-difference features negate under a swap — every `_DIFF` name is
`something(player1) - something(player2)`, so exchanging the players flips its sign. The two
match-context features, `BEST_OF` and `DRAW_SIZE`, describe the match rather than either player
and stay fixed. The swapped vector is therefore not `−x`, and even if it were, the model is a set
of axis-aligned thresholds fitted to data: nothing constrains it to satisfy
`f(swapped) == 1 − f(original)`. Scoring both orientations independently would produce two
probabilities that need not sum to 1 — two plausible but slightly inconsistent numbers, from a
system that should have one opinion about one match.

Fixing the internal order makes the two queries *the same computation*. Both directions descend the
same 25 trees over the same feature vector and read the same forest output; the query order only
decides which side of that single number is reported first. There is one opinion about the match,
and both phrasings of the question return it.

What the contract does and does not cover:

- Within one `Prediction`, `probability_a + probability_b == 1.0` holds exactly, because the second
  is computed as `1.0 - probability_a`.
- A separately reversed call reads the *same* forest score, but the mapping back to the requested
  order can apply another floating-point subtraction, so the mirrored value may differ in the last
  bit. Bit-identical reversal is therefore not part of the public contract; sharing one forest
  evaluation is.

The id-based rule is arbitrary but total: ids are unique integers, so the ordering is
deterministic, and `predict` refuses a player against themselves before reaching it.
`test_prediction_probabilities_are_complements` covers the within-answer complement;
`test_reversing_the_query_swaps_exactly` and
`test_the_internal_order_does_not_depend_on_the_query_order` check exact reversal for their
fixtures, which is evidence about those cases rather than a guarantee for every possible forest
score.

---

## 11. Compatibility and versioning

`BUNDLE_VERSION = 1` is the format version, added by the writer and required by the reader to match
exactly:

```python
if type(version) is not int or version != BUNDLE_VERSION:
    raise ValueError(f"bundle version {version!r} is not supported, expected {BUNDLE_VERSION}")
```

Not "at least", not "at most" — equal. There is one format, and this package reads it.

`package_version` records `matchmind_v1.__version__` at build time. It is required to be a non-empty
string and is carried for provenance; it does not gate loading. Compatibility is decided by things
that would actually change behaviour: the format version, the feature list, and the model parameter
contract.

**There is no migration path, and no attempt at one.** A bundle from a different format version,
or built for a different feature set, fails at load rather than being upgraded, partially read, or
loaded with defaults filled in. That is the right failure for this artifact: every alternative
produces a model that predicts confidently with silently shifted inputs. An artifact that cannot be
loaded correctly is rebuilt from source with `scripts/build_prediction_bundle.py`, which is cheap,
deterministic, and verified against the dataset hash.

---

## Related documentation

- [Architecture](architecture.md) — where the bundle sits between deployment build and prediction
- [Feature engineering](feature_engineering.md) — the feature order this format enforces
- [Model implementation](model_implementation.md) — the forest being serialized
- [Model evaluation](model_evaluation.md) — why this configuration was selected, and what the
  shipped model's numbers do and do not mean
