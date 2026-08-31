# Model Implementation

The from-scratch decision tree and random forest in `src/matchmind_v1/trees/`: how splits are
found, what the fast search actually does, and which behaviours are guaranteed rather than
incidental.

---

## 1. Scope

`trees/decision_tree.py` and `trees/random_forest.py` are the project's own binary classifiers.
`decision_tree.py` imports `math`, `dataclasses` and NumPy. `random_forest.py` imports NumPy and
the decision tree. Neither file imports scikit-learn, and neither delegates fitting or prediction
to anything but its own code.

scikit-learn appears in the project twice, both outside the models:

- `evaluation.py` imports four metric functions — `accuracy_score`, `log_loss`,
  `brier_score_loss`, `roc_auc_score`.
- `scripts/evaluate_models.py` fits `DecisionTreeClassifier` and `RandomForestClassifier`
  alongside the custom ones as reference implementations.

Those are the two. Under `src/`, `sklearn` appears in `evaluation.py` alone — not anywhere in
`trees/`.

The models are also entirely unaware of tennis: no import from `matchmind_v1.tennis` exists in
either file. They receive a float matrix and a binary vector.

---

## 2. The decision tree

### Node structure

One dataclass serves as both leaf and internal node:

```python
@dataclass
class _Node:
    prediction: int          # 0 or 1
    probability: float       # P(class 1) in this node
    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None
```

`feature is None` is the leaf test — there is no separate type and no `is_leaf` flag that could
disagree with the children. Every node carries a `prediction` and a `probability`, including
internal ones: `_grow` builds the leaf first and copies its values into the split node before
recursing. That makes the tree serializable as a uniform structure and means a node always knows
what it would answer if it were a leaf.

### Public interface

| Method | Behaviour |
|---|---|
| `fit(X, y)` | validates, resolves the candidate-feature count, seeds the RNG, grows the tree, returns `self` |
| `predict(X)` | `(predict_proba(X)[:, 1] >= 0.5).astype(int)` |
| `predict_proba(X)` | `(n_samples, 2)` array, column 1 is the leaf's `probability` |
| `score(X, y)` | mean accuracy |
| `get_depth()` | longest root-to-leaf path; a single-leaf tree is depth 0 |

`predict` is defined in terms of `predict_proba` rather than a separate descent, so the two can
never disagree about a sample. `test_predict_agrees_with_predict_proba` checks it anyway.

---

## 3. Input contract

Validation lives in four helpers shared with the forest, and it is strict on purpose: a silent
coercion here becomes a wrong model later.

**Feature matrices** (`_check_X`): complex input is rejected before anything else — including
complex arrays whose imaginary part happens to be zero, since accepting those would make the
rejection depend on the values rather than the dtype. The array is then cast to float, must be
2-D, must be non-empty, and must be entirely finite. NaN and infinity are refused rather than
propagated; at prediction time the column count must equal the fitted `n_features_`.

`allow_1d` lets `predict` and `predict_proba` take a single sample as a 1-D row, which is what the
prediction path passes. `fit` does not allow it — a 1-D `X` at fit time is far more likely to be a
mistake than an intent.

**Targets** (`_check_Xy`): a single-column 2-D `y` is raveled, then the values are checked *as
given*:

```python
values = y.astype(float)
if not np.isin(values, (0.0, 1.0)).all():
    raise ValueError("y must contain only the binary labels 0 and 1")
```

Checking before casting to int is the point. `np.asarray([0.2]).astype(int)` is `0`, so casting
first would silently accept fractional labels as class 0. `0.0` and `1.0` as floats are accepted,
since those are the same labels written differently.

**Constructor parameters**: `max_depth` and `min_samples_split` must be genuine integers —
`bool` is excluded explicitly, because `True` is an `int` in Python and `max_depth=True` is not a
depth of 1 that anyone meant. `min_impurity_decrease` must be a finite non-negative number,
`max_features` must be `None`, `"sqrt"` or a positive integer, and `random_state` must be `None`
or a non-negative integer.

---

## 4. Gini impurity

```python
def _gini(y):
    if y.size == 0:
        return 0.0
    p1 = float(np.mean(y))
    return 1.0 - p1**2 - (1.0 - p1) ** 2
```

For binary labels with `p₁` the fraction of class 1:

```
Gini = 1 − p₁² − (1 − p₁)²
```

Zero for a pure node, maximum 0.5 at `p₁ = 0.5`. Because the labels are `0`/`1` integers,
`np.mean(y)` *is* `p₁`, so no counting pass is needed.

A split is scored by the sample-weighted impurity of its two children:

```
weighted = (n_left · Gini(y_left) + n_right · Gini(y_right)) / n
decrease = Gini(parent) − weighted
```

The weighting is by node size, so a split that produces a pure child of two samples and a
50/50 child of two thousand is scored as the poor split it is.

---

## 5. Candidate thresholds

Candidate thresholds are the **midpoints between adjacent distinct sorted values** of a feature.
For a column whose sorted distinct values are `[1, 2, 5]`, the candidates are `1.5` and `3.5`.

The split rule is:

```
left:  feature <= threshold
right: feature >  threshold
```

Consistently, everywhere: in `_grow` when the mask is built, in `_descend` when a row is routed,
and in the reference implementation the tests compare against. A sample sitting exactly on a
threshold goes **left**, which `test_sample_equal_to_threshold_goes_left` pins with a single
column of `[0, 0, 2, 2]` — the threshold lands at `1.0`, and `1.0` predicts the left class.

Midpoints rather than observed values means the boundary is defined *between* two seen values
rather than on one of them. That is the mathematical intent; in floating point it is not a
guarantee, because the computed midpoint of two adjacent values can round onto the larger of them.
The implementation does not assume otherwise — rather than trusting the boundary index, it
recomputes the actual `<=` side with `np.searchsorted(values, thresholds, side="right")` and drops
any candidate whose left side turns out to be empty or to be everything. See "Midpoint rounding is
handled, not assumed away" in section 7.

---

## 6. The optimized split search

`_best_threshold` is the part of this implementation that does the real work, and it is what makes
fitting on ~90,000 rows take seconds.

### The obvious approach, and its cost

The straightforward reading of the split rule is: for each candidate threshold, build a boolean
mask over the column, use it to slice the labels, and count each side. That is one full pass over
the column *per threshold*. On the canonical data every continuous feature has close to as many
distinct values as rows, so a node with n samples evaluates roughly n thresholds and each costs
O(n) — the node's split search is quadratic in its own sample count, before multiplying by the
number of candidate features.

That version exists, in `tests/test_split_search.py`, as `ReferenceTree`. It is kept precisely
because it is the obvious reading, and the fast version must agree with it exactly.

### What the code does instead

```python
order = np.argsort(column, kind="stable")
values = column[order]
positives = np.cumsum(y[order])
n = values.size

boundary = np.flatnonzero(values[:-1] < values[1:])
if boundary.size == 0:
    return None
thresholds = (values[boundary] + values[boundary + 1]) / 2.0

n_left = np.searchsorted(values, thresholds, side="right")
usable = (n_left > 0) & (n_left < n)
if not usable.any():
    return None
thresholds, n_left = thresholds[usable], n_left[usable]

n_right = n - n_left
left_positive = positives[n_left - 1]
right_positive = positives[-1] - left_positive
p_left = left_positive / n_left
p_right = right_positive / n_right
weighted = (
    n_left * (1.0 - p_left**2 - (1.0 - p_left) ** 2)
    + n_right * (1.0 - p_right**2 - (1.0 - p_right) ** 2)
) / n
decrease = parent_impurity - weighted

first_best = int(np.argmax(decrease))
return float(thresholds[first_best]), float(decrease[first_best])
```

Step by step:

1. **Sort the column once**, stably, and keep the permutation.
2. **Reorder the labels** with the same permutation, so position in the sorted column and position
   in the label array line up.
3. **Cumulative sum of the labels.** Since labels are 0/1, `positives[i]` is the number of class-1
   samples among the first `i + 1` sorted samples.
4. **Find the boundaries** — indices where an adjacent pair of sorted values actually differ.
   Runs of ties produce no candidate, which is what makes an all-constant column return `None`.
5. **Compute the midpoints** at those boundaries, all at once.
6. **Determine the left-side size** for every threshold with one `searchsorted`.
7. **Read the class counts off the cumulative sum.** `left_positive = positives[n_left - 1]` — an
   array lookup, not a count. The right side is the total minus the left.
8. **Vectorize the weighted Gini** over every candidate at once, as array arithmetic.
9. **Take the best** with a single `argmax`.

The key move is step 7. Once the column is sorted, every candidate split is a *prefix* of the
sorted samples, and the number of class-1 labels in a prefix is a cumulative-sum lookup. No mask
is built, and the label array is never re-counted per threshold — after the sort, the entire
search over all candidates is a handful of whole-array operations.

### A small example

Column `[5, 1, 5, 2]` with labels `[1, 0, 1, 0]`:

```
sorted values      1    2    5    5
sorted labels      0    0    1    1
cumsum positives   0    0    1    2

boundaries between 1|2 and 2|5   ->  thresholds 1.5, 3.5
n_left  (searchsorted, right)    ->  1, 2
left_positive = positives[n_left-1] = positives[0], positives[1] = 0, 0
right_positive = 2 - 0 = 2,  2 - 0 = 2
```

At threshold 3.5: left is 2 samples with 0 positives (pure), right is 2 samples with 2 positives
(pure), weighted impurity 0, decrease 0.5. At 1.5: left 1 sample pure, right 3 samples with 2
positives, decrease is smaller. `argmax` picks 3.5.

### On complexity

The honest statement is the one the code supports: sorting the column dominates, and after it the
candidate scan does no per-threshold recounting. Both versions still sort or scan every column at
every node, so this is not an asymptotic change to the whole tree-building algorithm in every
regime — it removes the repeated full-column work that made the exhaustive version unusable on
this data. `test_fits_a_dataset_the_exhaustive_search_could_not` fits 20,000 rows × 20 continuous
columns, where the reference search evaluates close to 400,000 thresholds per node and took
minutes; the current one is comfortably under the test's deliberately loose 60-second bound.

---

## 7. Determinism and tie semantics

Several behaviours here are guaranteed rather than incidental, and the reference implementation in
the test suite has to reproduce every one of them exactly.

**The sort is stable.** `np.argsort(column, kind="stable")` is requested explicitly, which makes
the sorted representation of the column itself deterministic when values tie.

That determinism is not, however, something the split statistics depend on. Candidate thresholds
are only ever created *between distinct value groups*: `boundary = np.flatnonzero(values[:-1] <
values[1:])` skips every position inside a run of equal values. A valid split therefore always cuts
after a whole tied group, and `positives[n_left - 1]` sums the labels of every sample in that group
regardless of how they are arranged within it. Permuting labels inside an equal-value run changes
neither `n_left` nor the class counts on either side, so the chosen threshold and its impurity
decrease are unaffected.

The stable sort is a reproducibility choice about the intermediate array, not a correctness
dependency on tie order.

**Equally good thresholds resolve to the lowest.** `np.argmax` returns the *first* maximum, and
`thresholds` is in ascending order because it was built from a sorted column. So when several
thresholds tie on impurity decrease, the smallest one wins.

**Ties across features go to the first candidate examined.** `_best_split` replaces its incumbent
only on a strict improvement:

```python
if decrease > best_decrease:
```

So the first feature to achieve the best decrease keeps it. When `max_features` is `None`,
`_candidate_features` returns `np.arange(n_features)` and "first examined" means the lowest column
index — `test_identical_columns_resolve_to_the_same_one` duplicates column 0 into columns 3 and 5
and asserts the tree splits on column 0. When features are sampled, "first examined" means first in
the order the generator drew them, which is reproducible for a given `random_state` but is not
sorted by index.

**`min_impurity_decrease` is a strict floor.** `best_decrease` is *initialised* to
`self.min_impurity_decrease`, and the comparison is `>`. A split must beat the threshold, not
merely reach it. With the default `0.0`, a split with exactly zero decrease is refused, so a node
never splits into two children that are no better than itself.

**Midpoint rounding is handled, not assumed away.** A midpoint of two very close floats can round
onto the larger of the two values it came from. If that happens, the "left" side computed from the
boundary index would be wrong, so the code does not use the boundary index — it recomputes the left
count with `searchsorted(..., side="right")` and drops any candidate whose left side turns out to
be empty or to be everything. That is what the `usable` mask is for, and it is why an
all-constant column and a column with unrepresentable gaps both return `None` cleanly.

---

## 8. Growth and stopping

`_grow(X, y, depth)` builds the leaf first, then tries to justify splitting instead. There are five
ways to stop, checked in this order:

| # | Condition | Code |
|---|---|---|
| 1 | too few samples to split | `y.size < self.min_samples_split` |
| 2 | depth budget spent | `self.max_depth is not None and depth >= self.max_depth` |
| 3 | node is pure | `np.all(y == y[0])` |
| 4 | no usable split exists | `self._best_split(...) is None` — every candidate column constant |
| 5 | no split beats the impurity floor | also `None`, via the strict `>` in `_best_split` |

Conditions 4 and 5 both surface as `_best_split` returning `None`; they are distinct situations
with the same consequence.

Note that `min_samples_split` is checked against the node's own size, not its children's — there
is no `min_samples_leaf`. A qualifying split may produce a child of one sample.

When a split is taken, the mask is `X[:, feature] <= threshold` and recursion proceeds on
`X[mask]` / `X[~mask]`. Both children are grown eagerly; there is no pruning pass afterwards.

**Leaf values** come from `_make_leaf`:

```python
p1 = float(np.mean(y)) if y.size else 0.0
return _Node(prediction=int(p1 >= 0.5), probability=p1)
```

`probability` is the fraction of class-1 observations in the node — the training frequency, not a
smoothed or calibrated estimate. `prediction` is that fraction thresholded at 0.5, with an exact
tie of `p₁ == 0.5` resolving to class **1**. The same `>= 0.5` convention is used by `predict` at
the top level, so a leaf's stored `prediction` and what a single-tree `predict` returns for a
sample landing there always agree.

---

## 9. Feature subsampling

`_resolve_max_features` turns the constructor value into a count, once per fit:

| `max_features` | Candidates per split |
|---|---|
| `None` | all `n_features` |
| `"sqrt"` | `max(1, int(sqrt(n_features)))` — 8 of the 67 features |
| positive integer | `min(max_features, n_features)` |

`_candidate_features` then draws them per split:

```python
if self._n_candidates >= self.n_features_:
    return np.arange(self.n_features_)
return self._rng.choice(self.n_features_, size=self._n_candidates, replace=False)
```

Two details. Sampling is **without replacement**, so a split never wastes a slot on a duplicate
column. And when all features are candidates the function short-circuits to `arange` and consumes
no randomness at all — a tree with `max_features=None` draws nothing from its generator, which is
why a forest built with `max_features=None, bootstrap=False` produces identical trees
(`test_without_bootstrap_every_tree_sees_all_rows`).

`self._rng = np.random.default_rng(self.random_state)` is created in `fit`, not in `__init__`, so
the random stream restarts on every fit rather than continuing across them. With a fixed integer
`random_state`, each fit therefore builds the same generator from the same seed, and refitting on
the same data reproduces the same tree. With `random_state=None`, `default_rng` is seeded from the
operating system instead, so each fit draws a different stream and feature subsampling is
non-deterministic between runs.

---

## 10. The random forest

`RandomForest` bags decision trees. `fit` is short enough to read whole:

```python
X, y = _check_Xy(X, y)
self.n_features_ = X.shape[1]
self.estimators = []

rng = np.random.default_rng(self.random_state)
n_samples = X.shape[0]
seeds = rng.integers(0, 2**32 - 1, size=self.n_estimators)

for seed in seeds:
    if self.bootstrap:
        rows = rng.integers(0, n_samples, size=n_samples)
        X_sample, y_sample = X[rows], y[rows]
    else:
        X_sample, y_sample = X, y

    tree = DecisionTree(
        max_depth=self.max_depth,
        min_samples_split=self.min_samples_split,
        min_impurity_decrease=self.min_impurity_decrease,
        max_features=self.max_features,
        random_state=int(seed),
    )
    self.estimators.append(tree.fit(X_sample, y_sample))
```

**`fit` resets rather than accumulates.** `self.estimators = []` before the loop, so refitting a
forest gives `n_estimators` trees, not double that.
`test_refitting_resets_the_forest` covers it.

**Seeds are drawn up front, all at once**, before any bootstrap sampling — `n_estimators` integers
from `[0, 2³²−1)` in a single call. Each becomes one tree's `random_state`, which is what drives
that tree's per-split feature sampling. Drawing them in one block means a tree's feature sampling
depends only on its position in the forest, not on how much randomness earlier trees happened to
consume.

**Bootstrap samples come from the same generator**, inside the loop and after the seed block.
`rng.integers(0, n_samples, size=n_samples)` draws n row indices with replacement, so each tree
sees a resample of the same size as the training set, with roughly a third of rows absent and
others repeated. With `bootstrap=False` every tree gets the full `X` and `y` and the only
remaining source of diversity is feature subsampling.

**Everything else is forwarded unchanged.** `max_depth`, `min_samples_split`,
`min_impurity_decrease` and `max_features` are passed straight into each tree, and the tree's own
constructor validation is what enforces them — an invalid value raises from the forest's
constructor, not on the first fit.

**Prediction averages probabilities, not votes:**

```python
probabilities = np.stack([tree.predict_proba(X) for tree in self.estimators])
return probabilities.mean(axis=0)
```

Then `predict` thresholds the averaged `P(class 1)` at 0.5 — the same rule the tree uses.

Averaging leaf probabilities gives the forest a finer-grained score than a single leaf frequency:
25 trees can land between two leaves' values instead of on one of them. That is a statement about
the *resolution* of the output, not about its quality. Log loss and Brier score can then evaluate
how well that expressed confidence matches outcomes, rather than only judging which side of 0.5
the model fell on — but a finer-grained number earns a better score only if it is also better
aligned with what actually happened. Whether it is, for these models on this data, is measured in
[model_evaluation.md](model_evaluation.md), not asserted here.

`test_averaged_probabilities_are_smoother_than_a_single_tree` checks the visible consequence —
that the averaged output contains values strictly between 0 and 1 rather than collapsing to hard
0/1 leaves.

`n_estimators` defaults to 100 and `max_features` defaults to `"sqrt"`. The two sources of
randomness are independent: with `bootstrap=True` each tree receives its own bootstrap draw, and
feature subsampling adds per-split randomness on top. Either can make trees differ. Two
independently drawn bootstrap samples are not required to be distinct — nothing forbids the same
draw twice — so the accurate statement is that each tree gets its own draw, not that every draw
differs. How similar the trees would be with feature subsampling switched off is not something
this document measures.

The one case that is pinned down is the degenerate one: with `bootstrap=False` **and**
`max_features=None`, every tree receives identical data and consumes no split randomness at all,
and `test_without_bootstrap_every_tree_sees_all_rows` confirms the fixture's trees come out
identical.

---

## 11. Relationship to scikit-learn

scikit-learn is in this project as a **reference**, not as a dependency of the models.

It plays three roles, all of them external to `trees/`:

1. **Metrics.** `evaluation.py` uses its four scoring functions, so the numbers reported are
   computed by a standard implementation rather than a hand-rolled one.
2. **Reference models.** `scripts/evaluate_models.py` fits `DecisionTreeClassifier` and
   `RandomForestClassifier` on the identical splits with the identical candidate configurations.
   The comparison is between two implementations of the same idea, not between two different
   searches.
3. **Behavioural checks in the test suite.** `test_matches_sklearn_on_a_separable_problem` and
   `test_accuracy_is_close_to_sklearn_on_noisy_data` compare the custom tree against it on
   synthetic problems; the forest has its own equivalent.

The point of the comparison is not to beat scikit-learn, and not to match it exactly either.
Close agreement is useful evidence that the custom implementation behaves like a standard
CART/forest classifier on the same data and settings. A difference would be something to
investigate rather than automatic proof of a bug: two correct implementations can diverge over tie
handling, random-number streams, feature-sampling draws, threshold selection and bootstrap
samples, none of which are required to match across libraries.

The project's own behavioural contract comes from its implementation and its focused tests —
`ReferenceTree` for the optimized split search above all. scikit-learn is an independent sanity
comparison alongside that, not the specification. On the test period the two land within a
fraction of a point of each other; the numbers are in
[model_evaluation.md](model_evaluation.md#final-test-results).

The shipped model uses the custom forest with `n_estimators=25`, `max_depth=8`,
`min_samples_split=20`, `max_features="sqrt"`, `bootstrap=True`, `min_impurity_decrease=0.0`,
`random_state=0`. Those are the custom forest's own winning settings on the validation period —
the scikit-learn forest preferred 50 trees, but it is a reference implementation and does not
configure what ships. Why those settings were selected is in
[model_evaluation.md](model_evaluation.md).

---

## 12. Testing strategy

Four kinds of test carry most of the weight.

**The split-search oracle.** `tests/test_split_search.py` defines `ReferenceTree`, a subclass that
overrides `_best_split` with the slow mask-and-recount version, and then asserts the two trees
agree exactly. Concretely, a `structure()` helper renders each tree as nested tuples — `(feature,
threshold, left, right)` for a split, `(prediction, probability)` for a leaf — and the two
renderings must be equal, so every split feature, every threshold and every leaf probability
matches. On top of that it compares `get_depth()`, and `predict` and `predict_proba` outputs under
exact NumPy equality. (Nothing is serialized and no bytes are compared; the comparison is on the
in-memory structure and the outputs.)

That runs across five data shapes (continuous, heavily tied integers, constant columns, duplicated
columns, tiny) × six parameter settings × four seeds, plus a check that `min_impurity_decrease`
prunes identically at four cutoffs. The optimization is therefore not simply trusted to be
equivalent; it is tested against the obvious implementation on targeted cases chosen to exercise
ties, constant columns, duplicate columns, tiny inputs, several seeds and several impurity
cutoffs. That is a well-aimed suite rather than an exhaustive proof over every possible
floating-point matrix.

**Degenerate input validation.** Complex features, complex features with a zero imaginary part,
fractional labels, float-valued binary labels, mismatched lengths, wrong feature count at predict
time, 1-D input at fit time, NaN, predicting before fitting, bool-as-integer parameters,
non-finite `min_impurity_decrease`, negative `random_state`. Each has its own named test.

**Determinism and refit.** Same `random_state` produces identical predictions; different
`random_state` does not; refitting resets the estimator list; feature subsampling is reproducible.

**Probability shape and range.** `(n, 2)`, rows summing to 1, everything in `[0, 1]`, `predict`
agreeing with `predict_proba`, `score` matching a manually computed accuracy.

Alongside those sit structural tests that the tree learns what it should: it splits on the
informative feature of a separable problem, `max_depth=2` solves XOR, constant features produce a
single leaf, a pure target gives a zero-depth tree, and the forest generalizes at least as well as
one tree.

---

## Related documentation

- [Model evaluation](model_evaluation.md) — how these models were configured and scored
- [Architecture](architecture.md) — where the models sit in the system
- [Feature engineering](feature_engineering.md) — what the 67 columns mean
- [Prediction bundle format](artifact_format.md) — how a fitted forest is serialized
