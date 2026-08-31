"""Chronological splitting, model inputs and scoring for the canonical match dataset."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .data import TARGET_COLUMN
from .tennis import FEATURE_NAMES

# Fixed calendar periods, as YYYYMMDD bounds on tourney_date. Matches are predicted in
# the order they were played, so the model never sees a season it will be judged on.
#
# The periods are contiguous and stop at the end of 2025. The history runs on into 2026,
# but that season is only recorded as far as May, so scoring on it would measure a clay
# court stretch rather than a year. Those rows are outside every period on purpose and
# take no part in fitting, selection or scoring.
TRAIN_PERIOD = (19910101, 20211231)
VALIDATION_PERIOD = (20220101, 20231231)
TEST_PERIOD = (20240101, 20251231)

# A lower ATP rank number is the better player, so a negative difference favours player 1.
RANK_DIFF = FEATURE_NAMES.index("ATP_RANK_DIFF")


def split_by_period(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, validation and test, cut on tournament date and kept in order."""
    return tuple(
        dataset[dataset["tourney_date"].between(start, end)].reset_index(drop=True)
        for start, end in (TRAIN_PERIOD, VALIDATION_PERIOD, TEST_PERIOD)
    )


def features_and_target(matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Model inputs — exactly FEATURE_NAMES, in that order — and the binary target.

    Metadata columns identify a match; they are not evidence about who wins it, and
    selecting by name here is what keeps them out of the model.
    """
    X = matches[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = matches[TARGET_COLUMN].to_numpy(dtype=int)
    return X, y


def probability_scores(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Score predicted P(player 1 wins) four ways."""
    if not np.isfinite(probability).all():
        raise ValueError("predicted probabilities must be finite")
    if probability.min() < 0.0 or probability.max() > 1.0:
        raise ValueError("predicted probabilities must lie in [0, 1]")

    # Log loss is infinite at an exactly right or wrong certainty, so it gets the only
    # clipped copy. The other metrics score the model's actual output.
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return {
        "accuracy": float(accuracy_score(y_true, (probability >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Score a fitted classifier and time how long its predictions take."""
    started = time.perf_counter()
    probability = model.predict_proba(X)[:, 1]
    predict_seconds = time.perf_counter() - started
    return probability_scores(y, probability) | {"predict_seconds": predict_seconds}


def majority_label(y: np.ndarray) -> int:
    """The class a model would predict if it only knew how often each side wins."""
    return int(float(np.mean(y)) >= 0.5)


def rank_heuristic(X: np.ndarray) -> np.ndarray:
    """Back the better-ranked player. Exact rank ties go to player 2."""
    return (X[:, RANK_DIFF] < 0).astype(int)
