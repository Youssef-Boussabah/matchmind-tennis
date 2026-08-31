# Feature Engineering

The 67 model inputs: what each one measures, how it is computed, and what it does when there is
not enough history to compute it honestly.

Everything here comes from `src/matchmind_v1/tennis/features.py`, `state.py` and `elo.py`.

---

## 1. The core contract

Three rules hold for every feature without exception.

**Everything is pre-match.** A feature describes what was known *before* the match whose `RESULT`
sits on the same row. `build_features` reads `TennisState`; it never writes to it. The row is
finished before `state.update` is called, so a match cannot appear in its own history. See
[Data pipeline](data_pipeline.md) for where that ordering is enforced.

**Reads are pure and default-safe.** Every `TennisState` accessor is a `dict.get` with a default —
`elo_of` returns `DEFAULT_ELO`, `h2h_wins` returns `0`, `serve_history_of` returns an empty tuple.
Asking about a player who has never appeared returns a neutral value and inserts nothing, so
building features for a hypothetical matchup leaves the state exactly as it was.

**Orientation is player 1 minus player 2.** Every name ending in `_DIFF` is
`something(player1) - something(player2)`, which makes the feature set antisymmetric under a swap:
build the same matchup with the players exchanged and every difference feature negates.
`test_difference_features_are_antisymmetric_under_a_swap` checks twelve representatives spanning
every difference family.

### Reading a sign

What a positive value *means* is not the same question as whether it is *good*. Some of these
features measure an advantage; others just measure a quantity, and one measures a fault. Taking
them all as "favours player 1" would misread several of them.

| Feature(s) | A positive value means | Directional? |
|---|---|---|
| `ATP_RANK_DIFF` | player 1's rank **number** is larger, i.e. player 1 is the **worse**-ranked of the two | yes — **negative** is the better-ranked player |
| `ATP_POINTS_DIFF` | player 1 has more ranking points | yes — more is better |
| `ELO_DIFF`, `ELO_SURFACE_DIFF` | player 1 holds the higher rating | yes — higher is stronger |
| `H2H_DIFF`, `H2H_SURFACE_DIFF` | player 1 is ahead on the net head-to-head | yes — more wins over this opponent |
| `WIN_LAST_k_DIFF` | player 1 won more of their last k matches | yes — more wins |
| `ELO_GRAD_LAST_k_DIFF` | player 1's Elo slope is numerically higher — more positive **or** less negative | yes — higher slope, not necessarily rising |
| `P_ACE`, `P_1ST_IN`, `P_1ST_WON`, `P_2ND_WON`, `P_BP_SAVED` families | player 1 has the larger success rate on that serve statistic | yes — higher is better |
| `P_DF_LAST_k_DIFF` | player 1 double-faults more often | yes, but **adverse** — higher is worse |
| `AGE_DIFF` | player 1 is older | **no** — only that |
| `HEIGHT_DIFF` | player 1 is taller | **no** — only that |
| `N_GAMES_DIFF` | player 1 has more matches in the retained history | **no** — experience, not quality |

The rank feature is the one that most often gets misread, because the ATP's scale runs the wrong
way round: rank 1 is the best player, so the better player has the *smaller* number and a
**negative** difference. The rank baseline in [model evaluation](model_evaluation.md) is literally
`X[:, RANK_DIFF] < 0`.

`AGE_DIFF`, `HEIGHT_DIFF` and `N_GAMES_DIFF` are descriptive, not evaluative. `AGE_DIFF` positive
means older; `HEIGHT_DIFF` positive means taller; `N_GAMES_DIFF` counts matches retained in *this*
history, which reflects career length and data coverage as much as anything else. The feature
contract attaches no quality interpretation to any of them. The model may learn whatever
association the training data supports, including a non-monotonic one, and this document does not
assert what that association is.

Only `BEST_OF` and `DRAW_SIZE` are not differences. They describe the match, not either player,
and they are unchanged by a swap.

---

## 2. The inventory

`FEATURE_NAMES` is built by `_feature_names()` from three ingredients: a hand-written list of
eleven scalars, the `WINDOWS` tuple, and the `SERVE_FAMILIES` tuple.

```python
WINDOWS = (3, 5, 10, 25, 50, 100, 200)

SERVE_FAMILIES = (
    ("P_ACE", "ace"),
    ("P_DF", "double_fault"),
    ("P_1ST_IN", "first_serve_in"),
    ("P_1ST_WON", "first_serve_won"),
    ("P_2ND_WON", "second_serve_won"),
    ("P_BP_SAVED", "break_points_saved"),
)
```

| Family | Count | Shape |
|---|---:|---|
| Match context | 2 | one per attribute |
| Static player differences | 4 | one per attribute |
| Elo differences | 2 | overall and surface |
| Experience | 1 | matches played |
| Head-to-head | 2 | overall and surface |
| Elo trend | 7 | one per window |
| Recent wins | 7 | one per window |
| Serve statistics | 42 | 6 families × 7 windows |
| **Total** | **67** | |

`test_schema_has_sixty_seven_unique_features` asserts both the count and the uniqueness.

### Complete feature list

All 67 literals, in `FEATURE_NAMES` order, with the index each occupies in a feature vector. This
is the model's column contract: index 4 is `ATP_RANK_DIFF` in the canonical CSV, in
`features_and_target`, in every serialized tree's split, and in `feature_vector`.

```
match context and static differences
   0  BEST_OF
   1  DRAW_SIZE
   2  AGE_DIFF
   3  HEIGHT_DIFF
   4  ATP_RANK_DIFF
   5  ATP_POINTS_DIFF

Elo, experience and head-to-head
   6  ELO_DIFF
   7  ELO_SURFACE_DIFF
   8  N_GAMES_DIFF
   9  H2H_DIFF
  10  H2H_SURFACE_DIFF

Elo trend
  11  ELO_GRAD_LAST_3_DIFF
  12  ELO_GRAD_LAST_5_DIFF
  13  ELO_GRAD_LAST_10_DIFF
  14  ELO_GRAD_LAST_25_DIFF
  15  ELO_GRAD_LAST_50_DIFF
  16  ELO_GRAD_LAST_100_DIFF
  17  ELO_GRAD_LAST_200_DIFF

recent wins
  18  WIN_LAST_3_DIFF
  19  WIN_LAST_5_DIFF
  20  WIN_LAST_10_DIFF
  21  WIN_LAST_25_DIFF
  22  WIN_LAST_50_DIFF
  23  WIN_LAST_100_DIFF
  24  WIN_LAST_200_DIFF

serve rate: P_ACE  (state series 'ace')
  25  P_ACE_LAST_3_DIFF
  26  P_ACE_LAST_5_DIFF
  27  P_ACE_LAST_10_DIFF
  28  P_ACE_LAST_25_DIFF
  29  P_ACE_LAST_50_DIFF
  30  P_ACE_LAST_100_DIFF
  31  P_ACE_LAST_200_DIFF

serve rate: P_DF  (state series 'double_fault')
  32  P_DF_LAST_3_DIFF
  33  P_DF_LAST_5_DIFF
  34  P_DF_LAST_10_DIFF
  35  P_DF_LAST_25_DIFF
  36  P_DF_LAST_50_DIFF
  37  P_DF_LAST_100_DIFF
  38  P_DF_LAST_200_DIFF

serve rate: P_1ST_IN  (state series 'first_serve_in')
  39  P_1ST_IN_LAST_3_DIFF
  40  P_1ST_IN_LAST_5_DIFF
  41  P_1ST_IN_LAST_10_DIFF
  42  P_1ST_IN_LAST_25_DIFF
  43  P_1ST_IN_LAST_50_DIFF
  44  P_1ST_IN_LAST_100_DIFF
  45  P_1ST_IN_LAST_200_DIFF

serve rate: P_1ST_WON  (state series 'first_serve_won')
  46  P_1ST_WON_LAST_3_DIFF
  47  P_1ST_WON_LAST_5_DIFF
  48  P_1ST_WON_LAST_10_DIFF
  49  P_1ST_WON_LAST_25_DIFF
  50  P_1ST_WON_LAST_50_DIFF
  51  P_1ST_WON_LAST_100_DIFF
  52  P_1ST_WON_LAST_200_DIFF

serve rate: P_2ND_WON  (state series 'second_serve_won')
  53  P_2ND_WON_LAST_3_DIFF
  54  P_2ND_WON_LAST_5_DIFF
  55  P_2ND_WON_LAST_10_DIFF
  56  P_2ND_WON_LAST_25_DIFF
  57  P_2ND_WON_LAST_50_DIFF
  58  P_2ND_WON_LAST_100_DIFF
  59  P_2ND_WON_LAST_200_DIFF

serve rate: P_BP_SAVED  (state series 'break_points_saved')
  60  P_BP_SAVED_LAST_3_DIFF
  61  P_BP_SAVED_LAST_5_DIFF
  62  P_BP_SAVED_LAST_10_DIFF
  63  P_BP_SAVED_LAST_25_DIFF
  64  P_BP_SAVED_LAST_50_DIFF
  65  P_BP_SAVED_LAST_100_DIFF
  66  P_BP_SAVED_LAST_200_DIFF
```

Every windowed name follows `{PREFIX}_LAST_{k}_DIFF` for each `k` in `WINDOWS`. The overall
order is the eleven scalars, then the seven Elo trends, then the seven win counts, then each serve
family in `SERVE_FAMILIES` order across all seven windows.

---

## 3. Match context

`BEST_OF` is the match format as the source records it, 3 or 5 — `MatchContext` accepts nothing
else.

`DRAW_SIZE` is the source's draw-size value, an integer of at least 2. Not every draw size is a
power of two; the source contains 9, 10, 12, 18, 24 and 28.

Both are passed through unchanged. What either implies about a match is left to the model.

**Surface is not an integer feature, and that is deliberate.** There is no `SURFACE` column in
`FEATURE_NAMES`. Encoding "Clay = 2" would ask the model to learn an arbitrary numeric ordering of
four unordered categories. Instead, surface is used to *select which history to read*:

```python
"ELO_SURFACE_DIFF": state.surface_elo_of(surface, p1) - state.surface_elo_of(surface, p2),
"H2H_SURFACE_DIFF": state.surface_h2h_wins(surface, p1, p2) - state.surface_h2h_wins(surface, p2, p1),
```

Surface is therefore represented **indirectly**, through those two surface-specific historical
features, rather than through a categorical model column: it reaches the model as *what these two
players have done on this surface*. `test_surface_elo_difference_is_surface_specific` pins this
down.

That is a description of how this feature set works, not a claim that no other encoding could
help. A one-hot surface indicator, a player-by-surface preference term, or an interaction between
surface and serve statistics might all carry signal; none of them was built, so nothing here says
anything about how they would perform. Section 12 records the indirect representation as a
limitation.

---

## 4. Static player differences

Four features come from `PlayerSnapshot`, which holds the attributes recorded on the match row
itself — not averages, not historical state:

| Feature | Value |
|---|---|
| `AGE_DIFF` | `player1.age - player2.age`, in years as the source records them |
| `HEIGHT_DIFF` | `player1.height - player2.height`, in centimetres |
| `ATP_RANK_DIFF` | `player1.atp_rank - player2.atp_rank` — negative is the better-ranked player |
| `ATP_POINTS_DIFF` | `player1.atp_points - player2.atp_points` |

`build_features` draws its 67 values from three sources, and only the third is historical:

| Source | Features | Count |
|---|---|---:|
| `MatchContext` | `BEST_OF`, `DRAW_SIZE` | 2 |
| `PlayerSnapshot` (both players) | `AGE_DIFF`, `HEIGHT_DIFF`, `ATP_RANK_DIFF`, `ATP_POINTS_DIFF` | 4 |
| `TennisState` | everything else | 61 |

So six features never consult `TennisState`: the two match-context values and these four snapshot
differences. The snapshot four read the ATP's own assessment as of that match, which is part of
why the rank heuristic is such a strong baseline — the tour's ranking already encodes a season of
results.

At prediction time these come from the saved `PlayerProfile` — a player's **last observed**
attributes — which is a different thing from a current ranking and is called out in the README and
in [the bundle format](artifact_format.md#7-player-profiles).

---

## 5. Elo

`elo.py` is seventeen lines, and all of it is in use:

```python
DEFAULT_ELO = 1500.0
ELO_K = 24.0

expected_score(r, r_opp) = 1 / (1 + 10 ** ((r_opp - r) / 400))

shift = K * (1 - expected_score(winner, loser))
winner' = winner + shift
loser'  = loser  - shift
```

Standard Elo with a fixed K of 24 and the conventional 400-point scale, on which a 400-point lead
means a 10:1 expected score. The update is **zero-sum**: the winner gains exactly what the loser
loses, so the total rating across all players stays at `1500 × n_players` forever
(`test_update_conserves_total_rating`). The shift is largest when the result was unlikely and
approaches zero when it was expected, bounded above by K.

Two features read it:

- **`ELO_DIFF`** — `state.elo_of(p1) - state.elo_of(p2)`, one rating pool across all surfaces.
- **`ELO_SURFACE_DIFF`** — the same on a rating pool kept per surface. `TennisState.surface_elo`
  is a `dict[str, dict[int, float]]`, and `update` moves the played surface's ratings only.

**Unseen players.** `elo_of` and `surface_elo_of` return `DEFAULT_ELO` for a player with no
history, so two debutants have `ELO_DIFF == 0.0` — no information, correctly stated. A player with
history against a debutant is compared to the 1500 baseline rather than to nothing, which is the
honest default: it says "we know one of these two is above average" without inventing a rating for
the other. A player's surface Elo also starts at 1500 the first time they play on a new surface,
independent of how high their overall Elo has climbed.

---

## 6. Experience and head-to-head

| Feature | Definition |
|---|---|
| `N_GAMES_DIFF` | `matches_played_of(p1) - matches_played_of(p2)` — total matches in the replayed history, both players incremented on every update |
| `H2H_DIFF` | `h2h_wins(p1, p2) - h2h_wins(p2, p1)` — the net head-to-head between these two |
| `H2H_SURFACE_DIFF` | the same restricted to the surface being played |

`TennisState.h2h` is keyed by an ordered `(winner_id, loser_id)` pair, so `h2h_wins(a, b)` is
strictly "times a beat b". Taking the difference of both directions gives a signed net record:
`+2` means player 1 is two wins up, `0` means level *or* never met. The model cannot distinguish
those two cases from `H2H_DIFF` alone, though `N_GAMES_DIFF` and the window features carry the
general-experience signal separately.

`N_GAMES_DIFF` counts matches in *this* history — from 1991, and only rows that survived cleaning.
It is not a career total.

---

## 7. The rolling-window rule

Every `LAST_k` feature — 56 of the 67 — goes through one function:

```python
def _window_diff(history1, history2, k, summarise):
    if len(history1) < k or len(history2) < k:
        return 0.0
    return float(summarise(list(history1)[-k:]) - summarise(list(history2)[-k:]))
```

**A `LAST_k` feature is neutral `0.0` unless both players have at least k observations.** Not the
last min(k, available); not a partial average scaled up. Zero.

The reason is that `LAST_100` should mean a genuine hundred-match comparison. Without the rule, a
player with four recorded matches would have `WIN_LAST_3`, `WIN_LAST_5`, `WIN_LAST_10`,
`WIN_LAST_25`, `WIN_LAST_50`, `WIN_LAST_100` and `WIN_LAST_200` all reporting the same four-match
average, and seven columns that are supposed to describe seven different time horizons would carry
one number seven times. The model would then split on `LAST_200` as though it were long-run form
when it was in fact a four-match sample.

The requirement is on **both** players, because these are differences. Comparing player 1's genuine
last 100 against player 2's last 6 is not a hundred-match comparison either.

Two consequences worth knowing:

- `0.0` is overloaded. It means "no comparison available" and it also means "these two players are
  exactly level". The model cannot tell them apart. That is a real cost, accepted in exchange for
  the windows meaning what their names say.
- The windows fill in progressively. Early in a player's career only the short windows are live;
  the longer ones switch on as history accumulates. `WINDOWS` is `(3, 5, 10, 25, 50, 100, 200)` and
  `MAX_HISTORY = max(WINDOWS) = 200`, so the deques hold exactly enough for the largest window and
  `LAST_200` is satisfiable at the boundary.

`test_incomplete_windows_are_zero` and `test_windows_stay_zero_when_only_one_player_has_history`
cover both halves of the condition.

---

## 8. Elo trend

`ELO_GRAD_LAST_k_DIFF` compares the two players' Elo slopes.

`TennisState.elo_history[player]` is a deque of that player's rating **after** each of their
matches. `_window_diff` takes the last k values for each player and passes them to `_slope`:

```python
def _slope(values):
    return float(np.polyfit(np.arange(len(values)), np.asarray(values, dtype=float), 1)[0])
```

That is the gradient of an ordinary least-squares line fitted to the k ratings, and the feature is
`slope(player1) - slope(player2)` — a difference of two slopes, not the slope of a difference.

**The x-axis is observation index, not time.** `np.arange(len(values))` numbers the player's last
k matches 0, 1, 2, … k−1. The slope is therefore "Elo points gained per match played", not per day.
Two players with identical `ELO_GRAD_LAST_10_DIFF` may have accumulated those ten matches over six
weeks and over two years; the feature does not distinguish them, and it is not normalised for
elapsed time in any way. That is a genuine limitation, not a simplification of the description.

The units are rating points, so the magnitudes are meaningful: a player winning consistently gains
a few points per match, and the slope reflects it.
`test_elo_gradient_is_positive_for_a_rising_player` checks that a player on a winning run has a
positive slope.

**A positive difference does not mean player 1 is rising.** It means player 1's slope is
numerically higher than player 2's. Slopes of `-1` and `-2` give a difference of `+1` while both
players are declining — player 1 is simply declining less steeply. The feature orders the two
trends; it does not classify either as upward.

The feature reads the **Elo** deque, not the results deque, and the distinction matters.
`test_elo_gradient_uses_elo_history_not_results` constructs a player who wins three then loses
three: the last three results are a flat `[0, 0, 0]` with a slope of exactly zero, while the Elo
series over those same three matches is clearly falling. Elo carries opponent quality and the
magnitude of each swing; a binary result series carries neither.

---

## 9. Recent form

`WIN_LAST_k_DIFF` uses the same window machinery with `sum` as the summary function.

`TennisState.recent_results[player]` is a deque of integers: `update` appends `1` to the winner and
`0` to the loser of every match. So `sum` over the last k entries is literally **the number of wins
in that player's last k matches**, an integer in `[0, k]`, and the feature is
`wins1 - wins2` in `[-k, k]`.

`WIN_LAST_3_DIFF == 3` means player 1 won their last three and player 2 lost their last three. It
does not say anything about who those opponents were — that is what the Elo features are for. The
two describe different things and the model gets both.

---

## 10. Serve statistics

The largest family: 42 of the 67 features. Understanding them needs two steps, because what the
state stores is not what the source file records.

### Step one: counts become rates, once per match

`PlayerMatchStats` holds the eight raw counts from the source row. `TennisState.update` converts
them through `_serve_rates`, and stores **rates**, never counts. Each rate is a percentage on a
0–100 scale, and each has its own denominator:

| Family | Stat key | Formula |
|---|---|---|
| `P_ACE` | `ace` | `100 × ace / service_points` |
| `P_DF` | `double_fault` | `100 × double_fault / service_points` |
| `P_1ST_IN` | `first_serve_in` | `100 × first_serve_in / service_points` |
| `P_1ST_WON` | `first_serve_won` | `100 × first_serve_won / first_serve_in` |
| `P_2ND_WON` | `second_serve_won` | `100 × second_serve_won / (service_points − first_serve_in)` |
| `P_BP_SAVED` | `break_points_saved` | `100 × break_points_saved / break_points_faced` |

The denominators are the point of the exercise. `P_1ST_WON` is *first serves won out of first
serves landed*, not out of all service points — a player who lands 50% of first serves and wins 80%
of those is a different player from one who lands 80% and wins 50%, and dividing both by
`service_points` would blur them together. `P_2ND_WON` divides by second serves *played*, which is
`service_points − first_serve_in` and is never stored in the source; it is derived. `P_BP_SAVED`
divides by break points faced, which is unrelated to serve volume.

The `PlayerMatchStats` arithmetic checks exist to protect exactly these divisions: if
`first_serve_in > service_points` were allowed through, `P_1ST_IN` would exceed 100% and every
downstream average would be quietly wrong.

**A zero denominator records nothing.** `_serve_rates` omits a stat rather than storing `0.0`:

```python
if stats.break_points_faced > 0:
    rates["break_points_saved"] = 100.0 * stats.break_points_saved / stats.break_points_faced
```

A player who faced no break points did not save 0% of them — there is nothing to record, and
storing a zero would drag their break-point average down for the next 200 matches. The observation
simply does not happen. `test_zero_denominators_record_nothing_rather_than_a_placeholder` holds
this.

The consequence is that **the six serve deques for one player can have different lengths**.
`serve_history[player]` is a `dict[str, deque]`, one deque per stat, each capped at 200. A player
with 200 recorded `ace` rates may have only 180 `break_points_saved` rates, because in twenty of
those matches they faced no break point. Each family's window requirement is evaluated against its
own deque, so `P_ACE_LAST_200_DIFF` can be live while `P_BP_SAVED_LAST_200_DIFF` is still neutral.

### Step two: rates become windowed differences

For each family and each window k, using `fmean` as the summary:

```
mean of player 1's last k rates  −  mean of player 2's last k rates
```

subject to the full-window rule from section 7 — neutral `0.0` unless both players have at least k
recorded rates for **that specific statistic**.

So `P_1ST_WON_LAST_25_DIFF` is: the mean of player 1's first-serve-won percentage over their last
25 matches in which they landed at least one first serve, minus the same for player 2, in
percentage points. A value of `+3.5` means player 1 has been winning three and a half percentage
points more of their first-serve points over that stretch.

This is a mean of per-match percentages, not a pooled percentage over aggregated counts. A match
with 40 service points weighs the same as one with 140. That is a modelling choice, not an
oversight: it treats each match as one observation of a player's serving level, which is what the
rolling-window framing is measuring.

---

## 11. Why the order is fixed

`FEATURE_NAMES` is a module-level `tuple`, and `feature_vector` follows it explicitly:

```python
def feature_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)
```

`build_features` returns a `dict`, which is convenient to read and to test by name. But a tree
model does not learn feature *names* — it learns "column 4 splits at −11.5". Feed it the same
features in a different column order and every threshold in every node now refers to the wrong
quantity. The model will not error; it will produce confident nonsense.

So the dict is never the contract. The tuple is. Three places follow it, and they must agree:

| Place | How |
|---|---|
| Dataset build | `[features[name] for name in FEATURE_NAMES]` in `build_dataset` |
| Evaluation | `matches[list(FEATURE_NAMES)].to_numpy()` in `features_and_target` |
| Prediction | `feature_vector(features)` in `PredictionBundle.predict` |

The serialized bundle stores the order too, as `metadata.feature_names`, and `_check_metadata`
refuses to load a bundle whose list is not equal to `FEATURE_NAMES` — element for element, in
order. A bundle built against a different feature set fails at load rather than predicting with
shifted columns. See [Prediction bundle format](artifact_format.md#3-metadata-contract).

Two tests hold the contract: `test_build_features_matches_the_schema_exactly` (the dict's keys are
exactly the tuple's entries) and `test_feature_vector_follows_schema_order`.

---

## 12. Limitations

Specific to the feature set. The evaluation-level limitations are in
[model_evaluation.md](model_evaluation.md#limitations).

- **No time-since-last-match, and no time-based decay.** `LAST_10` is a player's last ten matches
  whether they were played across two months or two years. Waiting does not change a player's
  stored state, and no feature measures elapsed time, so two otherwise identical states separated
  by different calendar gaps produce the same history-derived features. (A returning player does
  still carry their Elo, surface Elo, match count, head-to-head and rolling histories — what is
  missing is the gap itself.)
- **The Elo trend is per match, not per day.** `_slope` fits against observation index. See
  section 8.
- **Surface is indirect.** It reaches the model only through `ELO_SURFACE_DIFF` and
  `H2H_SURFACE_DIFF`. A player's general surface preference, beyond what their surface Elo has
  accumulated, is not represented.
- **`0.0` is ambiguous.** For a windowed feature it means either "not enough history" or "exactly
  level".
- **Missing raw statistics reject the row.** There is no feature-level imputation anywhere; a match
  with unusable serve counts never becomes a row at all. See
  [Data pipeline](data_pipeline.md#3-validation-and-rejection).
- **Serve rates are per-match means.** A short match counts as much as a long one within a window.
- **No opponent-adjusted serve statistics.** `P_ACE_LAST_10_DIFF` does not know whether those ten
  matches were against top-10 returners.
- **Model outputs are not calibrated probabilities.** The features produce a score that ranks
  matchups; nothing here fits a calibration map.

---

## Related documentation

- [Architecture](architecture.md) — the state/feature boundary these rules depend on
- [Data pipeline](data_pipeline.md) — how a match reaches `build_features`
- [Model evaluation](model_evaluation.md) — how much these features are actually worth
- [Prediction bundle format](artifact_format.md) — how the feature order is enforced at load time
