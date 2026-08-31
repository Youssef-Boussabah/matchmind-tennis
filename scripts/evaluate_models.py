"""Train and evaluate the tree models on the canonical match dataset.

Configurations are chosen on the validation period alone. The test period is scored
once, at the end, with every choice already frozen.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from matchmind_v1.data import check_dataset
from matchmind_v1.evaluation import (
    TEST_PERIOD,
    TRAIN_PERIOD,
    VALIDATION_PERIOD,
    evaluate,
    features_and_target,
    majority_label,
    rank_heuristic,
    split_by_period,
)
from matchmind_v1.tennis import FEATURE_NAMES
from matchmind_v1.trees import DecisionTree, RandomForest

DEFAULT_DATASET = Path("data/processed/match_features.csv")
DEFAULT_METRICS = Path("results/test_metrics.csv")
DEFAULT_VALIDATION = Path("results/validation_results.csv")

TREE_SETTINGS = [
    {"max_depth": 4, "min_samples_split": 20},
    {"max_depth": 6, "min_samples_split": 20},
    {"max_depth": 8, "min_samples_split": 20},
    {"max_depth": 8, "min_samples_split": 50},
]
FOREST_SETTINGS = [
    {"n_estimators": 25, "max_depth": 6, "min_samples_split": 20},
    {"n_estimators": 25, "max_depth": 8, "min_samples_split": 20},
    {"n_estimators": 50, "max_depth": 8, "min_samples_split": 20},
]

# The same settings are offered to both implementations, so the comparison is between
# the implementations rather than between two different searches.
CANDIDATES = [
    ("decision tree", "custom", partial(DecisionTree, random_state=0), TREE_SETTINGS),
    ("decision tree", "sklearn", partial(DecisionTreeClassifier, random_state=0), TREE_SETTINGS),
    (
        "random forest",
        "custom",
        partial(RandomForest, max_features="sqrt", bootstrap=True, random_state=0),
        FOREST_SETTINGS,
    ),
    (
        "random forest",
        "sklearn",
        partial(RandomForestClassifier, max_features="sqrt", bootstrap=True, random_state=0),
        FOREST_SETTINGS,
    ),
]

METRIC_COLUMNS = ["accuracy", "log_loss", "brier", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--validation-results", type=Path, default=DEFAULT_VALIDATION)
    return parser.parse_args()


def fit_and_score(make, params, X_fit, y_fit, X_score, y_score):
    """Fit one configuration and score it, returning the model and its scores."""
    model = make(**params)
    started = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started
    return model, evaluate(model, X_score, y_score) | {"fit_seconds": fit_seconds}


def score_candidates(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Every candidate configuration, fitted on train and scored on validation."""
    X_train, y_train = features_and_target(train)
    X_validation, y_validation = features_and_target(validation)

    baseline = majority_label(y_train)
    rows = [
        {
            "family": "majority baseline",
            "implementation": "-",
            "parameters": json.dumps({"label": baseline}),
            "accuracy": round(float((y_validation == baseline).mean()), 6),
        },
        {
            "family": "rank heuristic",
            "implementation": "-",
            "parameters": json.dumps({"rule": "lower ATP rank wins, ties to player 2"}),
            "accuracy": round(float((y_validation == rank_heuristic(X_validation)).mean()), 6),
        },
    ]
    for family, implementation, make, settings in CANDIDATES:
        for params in settings:
            _, scores = fit_and_score(make, params, X_train, y_train, X_validation, y_validation)
            row = {
                "family": family,
                "implementation": implementation,
                "parameters": json.dumps(params, sort_keys=True),
                "fit_seconds": round(scores["fit_seconds"], 2),
                **{name: round(scores[name], 6) for name in METRIC_COLUMNS},
            }
            rows.append(row)
            print(
                f"  {implementation:>7} {family:<14} {row['parameters']:<58}"
                f" log loss {row['log_loss']:.4f}  acc {row['accuracy']:.4f}"
                f"  {row['fit_seconds']:>6.2f}s"
            )
    columns = ["family", "implementation", "parameters", "fit_seconds", *METRIC_COLUMNS]
    return pd.DataFrame(rows)[columns]


def chosen_settings(candidates: pd.DataFrame, family: str, implementation: str) -> dict:
    """The configuration with the lowest validation log loss."""
    rows = candidates[
        (candidates["family"] == family) & (candidates["implementation"] == implementation)
    ]
    return json.loads(rows.loc[rows["log_loss"].idxmin(), "parameters"])


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    dataset = pd.read_csv(args.dataset, low_memory=False)
    check_dataset(dataset)
    train, validation, test = split_by_period(dataset)

    # The periods stop at the end of 2025; the dataset runs into a partial 2026. Those
    # later rows are excluded on purpose, so the three periods have to account for every
    # row inside the window rather than for the whole dataset.
    eligible = dataset["tourney_date"].between(TRAIN_PERIOD[0], TEST_PERIOD[1])
    excluded = dataset[~eligible]

    print(f"{len(dataset):,} matches from {args.dataset}")
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        dates = part["tourney_date"]
        print(f"  {name:<11} {len(part):>7,} rows  {dates.min()} to {dates.max()}")
    print(f"  {'eligible':<11} {int(eligible.sum()):>7,} rows"
          f"  {TRAIN_PERIOD[0]} to {TEST_PERIOD[1]}")
    if len(excluded):
        dates = excluded["tourney_date"]
        print(f"  {'excluded':<11} {len(excluded):>7,} rows  {dates.min()} to {dates.max()}"
              f"  (outside the evaluation window, incomplete season)")

    # Together these say the three periods tile the window: nothing eligible falls into a
    # gap between them, and nothing is counted twice by an overlap.
    covered = (
        dataset["tourney_date"].between(*TRAIN_PERIOD)
        | dataset["tourney_date"].between(*VALIDATION_PERIOD)
        | dataset["tourney_date"].between(*TEST_PERIOD)
    )
    if not covered.equals(eligible):
        raise ValueError("the chronological periods leave a gap inside the evaluation window")
    if len(train) + len(validation) + len(test) != int(eligible.sum()):
        raise ValueError("the chronological periods do not cover the evaluation window exactly")

    print("\nvalidation:")
    candidates = score_candidates(train, validation)
    args.validation_results.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.validation_results, index=False)

    # Selection is over. From here the models are refitted on all eligible history
    # through the end of 2023
    # and the test period is scored once.
    fitting = pd.concat([train, validation], ignore_index=True)
    X_fit, y_fit = features_and_target(fitting)
    X_test, y_test = features_and_target(test)

    baseline = majority_label(y_fit)
    rows = [
        {
            "model": "majority baseline",
            "implementation": "-",
            "parameters": json.dumps({"label": baseline}),
            "accuracy": float((y_test == baseline).mean()),
        },
        {
            "model": "rank heuristic",
            "implementation": "-",
            "parameters": json.dumps({"rule": "lower ATP rank wins, ties to player 2"}),
            "accuracy": float((y_test == rank_heuristic(X_test)).mean()),
        },
    ]

    print("\ntest:")
    for family, implementation, make, _ in CANDIDATES:
        params = chosen_settings(candidates, family, implementation)
        model, scores = fit_and_score(make, params, X_fit, y_fit, X_test, y_test)
        rows.append(
            {
                "model": family,
                "implementation": implementation,
                "parameters": json.dumps(params, sort_keys=True),
                "fit_seconds": round(scores["fit_seconds"], 2),
                "predict_seconds": round(scores["predict_seconds"], 3),
                **{name: scores[name] for name in METRIC_COLUMNS},
            }
        )
        print(
            f"  {implementation:>7} {family:<14} acc {scores['accuracy']:.4f}"
            f"  log loss {scores['log_loss']:.4f}  brier {scores['brier']:.4f}"
            f"  auc {scores['roc_auc']:.4f}"
        )
        if implementation == "sklearn":
            report_importances(model, family)

    metrics = pd.DataFrame(rows)
    metrics.insert(2, "split", "test")
    metrics.insert(3, "n_rows", len(test))
    metrics = metrics[
        ["model", "implementation", "split", "n_rows", *METRIC_COLUMNS,
         "fit_seconds", "predict_seconds", "parameters"]
    ]
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.round(6).to_csv(args.metrics, index=False)

    print(f"\nwrote {args.validation_results} and {args.metrics}")
    print(f"finished in {time.perf_counter() - started:.1f}s")


def report_importances(model, family: str, top: int = 10) -> None:
    """sklearn's impurity-based feature importances, as a diagnostic only."""
    ranked = pd.Series(model.feature_importances_, index=FEATURE_NAMES).nlargest(top)
    print(f"    top {top} impurity-based importances, sklearn {family}:")
    for name, weight in ranked.items():
        print(f"      {name:<28} {weight:.4f}")


if __name__ == "__main__":
    main()
