"""Train the deployment model on all canonical history and save the prediction bundle.

This is not evaluation. The configuration was chosen and scored on held-out seasons
elsewhere; here it is refitted on every canonical match so the saved model has seen the
most recent history it can before being asked about a future one.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from matchmind_v1 import __version__
from matchmind_v1.artifacts import load_bundle, save_bundle
from matchmind_v1.data import (
    FIRST_YEAR,
    LAST_YEAR,
    check_dataset,
    latest_snapshots,
    load_matches,
    load_player_names,
    prepare_matches,
    replay_state,
)
from matchmind_v1.evaluation import features_and_target
from matchmind_v1.prediction import PlayerProfile, PredictionBundle
from matchmind_v1.tennis import FEATURE_NAMES, MatchContext, build_features
from matchmind_v1.trees import RandomForest

DEFAULT_DATASET = Path("data/processed/match_features.csv")
DEFAULT_PLAYERS = Path("data/atp_players.csv")
DEFAULT_RAW = Path("data/all")
DEFAULT_OUTPUT = Path("models/matchmind_v1_bundle.json.gz")

# The dataset this model is meant to be trained on. Refusing anything else keeps a
# stale or hand-edited feature file from quietly becoming the shipped model.
EXPECTED_DATASET_SHA256 = "22425a372212c475ff0eacaf7a85916cc2468d16a504641ad06883754ddbe1ef"

# Chosen on the 2022-2023 validation seasons, by lowest validation log loss, and frozen
# before the test period was scored. These are the custom RandomForest's own winning
# settings: the scikit-learn forest preferred 50 trees, but it is a reference
# implementation and does not configure the model that ships.
MODEL_SETTINGS = {
    "n_estimators": 25,
    "max_depth": 8,
    "min_samples_split": 20,
    "min_impurity_decrease": 0.0,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--players", type=Path, default=DEFAULT_PLAYERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    digest = sha256_of(args.dataset)
    if digest != EXPECTED_DATASET_SHA256:
        raise SystemExit(
            f"{args.dataset} has SHA-256 {digest},\n"
            f"expected {EXPECTED_DATASET_SHA256}.\n"
            "Rebuild it with scripts/build_dataset.py before training the deployment model."
        )

    dataset = pd.read_csv(args.dataset, low_memory=False)
    check_dataset(dataset)
    print(f"{len(dataset):,} canonical matches, SHA-256 verified")

    X, y = features_and_target(dataset)
    model = RandomForest(**MODEL_SETTINGS)
    fit_started = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - fit_started
    print(f"trained {model.n_estimators} trees on all {len(X):,} rows in {fit_seconds:.1f}s")

    matches = prepare_matches(load_matches(args.raw_dir, FIRST_YEAR, LAST_YEAR))
    if len(matches) != len(dataset):
        raise SystemExit(f"raw history has {len(matches):,} rows, dataset has {len(dataset):,}")

    state = replay_state(matches)
    history_end_date = int(matches["tourney_date"].iloc[-1])
    print(f"replayed history to {history_end_date}, {len(state.elo):,} players rated")

    names = load_player_names(args.players)
    profiles = {}
    for player_id, (snapshot, last_seen) in latest_snapshots(matches).items():
        profiles[player_id] = PlayerProfile(
            snapshot=snapshot, last_seen_date=last_seen, name=names.get(player_id)
        )
    named = sum(1 for profile in profiles.values() if profile.name)
    print(f"{len(profiles):,} profiles, {named:,} named, {len(profiles) - named:,} id-only")

    bundle = PredictionBundle(
        model=model,
        state=state,
        profiles=profiles,
        metadata={
            "package_version": __version__,
            "dataset_rows": len(dataset),
            "dataset_sha256": digest,
            "feature_names": list(FEATURE_NAMES),
            "model_parameters": dict(MODEL_SETTINGS),
            "history_start_date": int(matches["tourney_date"].iloc[0]),
            "history_end_date": history_end_date,
            "profile_count": len(profiles),
        },
    )

    save_bundle(bundle, args.output)
    reloaded = load_bundle(args.output)
    check_round_trip(bundle, reloaded, X)

    size = args.output.stat().st_size
    print(f"\n{args.output}  {size:,} bytes")
    print(f"SHA-256 {sha256_of(args.output)}")
    print(f"{len(reloaded.model.estimators)} trees, {len(FEATURE_NAMES)} features")
    print(f"built in {time.perf_counter() - started:.1f}s")


def check_round_trip(bundle: PredictionBundle, reloaded: PredictionBundle, X: np.ndarray) -> None:
    """The reloaded bundle has to behave exactly like the one that was saved."""
    sample = X[:: max(1, len(X) // 500)]
    if not np.array_equal(bundle.model.predict_proba(sample), reloaded.model.predict_proba(sample)):
        raise SystemExit("reloaded model gives different probabilities")
    if not np.array_equal(bundle.model.predict(sample), reloaded.model.predict(sample)):
        raise SystemExit("reloaded model gives different predictions")

    if bundle.profiles.keys() != reloaded.profiles.keys():
        raise SystemExit("reloaded bundle has different players")
    for player_id, profile in bundle.profiles.items():
        if reloaded.profiles[player_id] != profile:
            raise SystemExit(f"profile for {player_id} changed on reload")

    original, restored = bundle.state, reloaded.state
    if original.elo != restored.elo or original.matches_played != restored.matches_played:
        raise SystemExit("reloaded state has different Elo or match counts")
    if original.h2h != restored.h2h or original.surface_elo != restored.surface_elo:
        raise SystemExit("reloaded state has different head-to-head or surface Elo")
    for player_id in list(original.elo)[:200]:
        if list(original.elo_history_of(player_id)) != list(restored.elo_history_of(player_id)):
            raise SystemExit(f"Elo history for {player_id} changed on reload")
        for stat in ("ace", "first_serve_in", "break_points_saved"):
            if list(original.serve_history_of(player_id, stat)) != list(
                restored.serve_history_of(player_id, stat)
            ):
                raise SystemExit(f"serve history for {player_id} changed on reload")

    context = MatchContext("Hard", 3, 128)
    players = list(bundle.profiles)
    for first, second in zip(players[:50], players[50:100], strict=True):
        before = build_features(
            bundle.profiles[first].snapshot, bundle.profiles[second].snapshot, context, original
        )
        after = build_features(
            reloaded.profiles[first].snapshot,
            reloaded.profiles[second].snapshot,
            context,
            restored,
        )
        if before != after:
            raise SystemExit(f"features for {first} against {second} changed on reload")

    print("round trip verified: model, state, profiles and features all match")


if __name__ == "__main__":
    main()
