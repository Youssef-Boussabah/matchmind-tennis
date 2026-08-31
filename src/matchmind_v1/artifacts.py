"""Saving and loading a prediction bundle as deterministic gzipped JSON."""

from __future__ import annotations

import gzip
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from .prediction import PlayerProfile, PredictionBundle
from .tennis import FEATURE_NAMES, SURFACES, PlayerSnapshot, TennisState
from .tennis.state import MAX_HISTORY, SERVE_STATS
from .trees import RandomForest
from .trees.decision_tree import DecisionTree, _Node

BUNDLE_VERSION = 1

# What a saved node needs to be a split rather than a leaf.
SPLIT_KEYS = ("feature", "threshold", "left", "right")

MODEL_PARAMETERS = (
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_impurity_decrease",
    "max_features",
    "bootstrap",
    "random_state",
)

# Serve percentages are stored on a 0-100 scale.
MAX_SERVE_RATE = 100.0


def save_bundle(bundle: PredictionBundle, path: str | Path) -> None:
    """Write the bundle. The same inputs always produce the same bytes."""
    payload = {
        "metadata": dict(bundle.metadata) | {"bundle_version": BUNDLE_VERSION},
        "model": _forest_to_json(bundle.model),
        "state": _state_to_json(bundle.state),
        "profiles": _profiles_to_json(bundle.profiles),
    }
    # allow_nan=False: a NaN would be written as the JSON-invalid literal NaN, and a
    # model that produced one has no business being saved.
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 and an empty filename keep the build clock and the path out of the header.
    with open(path, "wb") as raw, gzip.GzipFile(
        fileobj=raw, mode="wb", compresslevel=9, filename="", mtime=0
    ) as compressed:
        compressed.write(text.encode("utf-8"))


def load_bundle(path: str | Path) -> PredictionBundle:
    """Read a bundle back, rejecting anything this package cannot safely predict with.

    Everything here treats the file as untrusted input. It may have been edited, half
    written, or produced by another version, so nothing is coerced into the type it was
    supposed to have: a value that is not already right is an error.
    """
    try:
        with gzip.open(path, "rb") as compressed:
            payload = json.loads(
                compressed.read().decode("utf-8"),
                parse_float=_parse_float,
                parse_constant=_reject_constant,
            )
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not a readable prediction bundle: {error}") from error

    try:
        bundle = _bundle_from_payload(payload)
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError(f"{path} is not a well-formed prediction bundle: {error}") from error

    _check_predicts(bundle)
    return bundle


def _bundle_from_payload(payload: dict) -> PredictionBundle:
    metadata = payload["metadata"]
    _check_metadata(metadata)

    model = _forest_from_json(payload["model"], metadata["model_parameters"])
    if len(model.estimators) != model.n_estimators:
        raise ValueError(
            f"bundle holds {len(model.estimators)} trees, "
            f"its own settings say {model.n_estimators}"
        )

    profiles = _profiles_from_json(payload["profiles"])
    if not profiles:
        raise ValueError("bundle holds no player profiles")
    if metadata["profile_count"] != len(profiles):
        raise ValueError(
            f"bundle holds {len(profiles)} profiles, metadata says {metadata['profile_count']}"
        )

    state = _state_from_json(payload["state"])
    unrepresented = sorted(set(state.elo) - set(profiles))
    if unrepresented:
        raise ValueError(
            f"{len(unrepresented)} rated players have no profile, "
            f"starting with {unrepresented[0]}"
        )

    return PredictionBundle(model=model, state=state, profiles=profiles, metadata=metadata)


def _parse_float(text: str) -> float:
    """1e999 is valid JSON syntax and parses to infinity, which nothing here may hold."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"bundle contains the out-of-range number {text}")
    return value


def _reject_constant(name: str):
    """Python's JSON reader accepts NaN and Infinity; nothing here may contain them."""
    raise ValueError(f"bundle contains the non-finite value {name}")


def _json_int(value, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """A JSON integer. A float that happens to be whole is not one, and neither is a bool."""
    if type(value) is not int:
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return value


def _json_number(
    value, name: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    """A finite JSON number. Strings are not numbers here, however numeric they look."""
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}, got {value}")
    return float(value)


def _json_text(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _check_metadata(metadata) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("bundle metadata must be an object")

    version = metadata.get("bundle_version")
    if type(version) is not int or version != BUNDLE_VERSION:
        raise ValueError(f"bundle version {version!r} is not supported, expected {BUNDLE_VERSION}")

    names = metadata.get("feature_names")
    if not isinstance(names, list) or tuple(names) != FEATURE_NAMES:
        raise ValueError("bundle was built for a different feature set")
    if not isinstance(metadata.get("model_parameters"), dict):
        raise ValueError("metadata model_parameters must be an object")

    _json_text(metadata.get("dataset_sha256"), "dataset_sha256")
    _json_text(metadata.get("package_version"), "package_version")
    _json_int(metadata.get("dataset_rows"), "dataset_rows", minimum=1)
    _json_int(metadata.get("profile_count"), "profile_count", minimum=1)
    start = _json_int(metadata.get("history_start_date"), "history_start_date")
    end = _json_int(metadata.get("history_end_date"), "history_end_date")
    if start > end:
        raise ValueError(f"history starts at {start}, after it ends at {end}")


def _check_predicts(bundle: PredictionBundle) -> None:
    """One cheap prediction, to be sure the loaded model can produce a probability."""
    probabilities = bundle.model.predict_proba(np.zeros((1, len(FEATURE_NAMES))))
    if probabilities.shape != (1, 2):
        raise ValueError("loaded model does not return a pair of class probabilities")
    if not np.isfinite(probabilities).all() or probabilities.min() < 0.0:
        raise ValueError("loaded model produces invalid probabilities")
    if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("loaded model probabilities do not sum to one")


def _forest_to_json(forest: RandomForest) -> dict:
    return {
        "n_features": forest.n_features_,
        "parameters": {name: getattr(forest, name) for name in MODEL_PARAMETERS},
        "trees": [_tree_to_json(tree) for tree in forest.estimators],
    }


def _forest_from_json(data: dict, declared: dict) -> RandomForest:
    forest = RandomForest(**_forest_parameters(data["parameters"], declared))
    forest.n_features_ = _json_int(data["n_features"], "model n_features", minimum=1)
    if forest.n_features_ != len(FEATURE_NAMES):
        raise ValueError(
            f"model takes {forest.n_features_} features, expected {len(FEATURE_NAMES)}"
        )

    trees = data["trees"]
    if not isinstance(trees, list) or not trees:
        raise ValueError("bundle holds no trees")
    forest.estimators = [_tree_from_json(tree, forest.n_features_) for tree in trees]
    return forest


def _forest_parameters(stored, declared) -> dict:
    """The settings the forest is rebuilt from, required in full in both copies.

    Every parameter has to be there: a missing one would be filled in from the
    constructor's default, quietly producing a different model from the saved one. The
    two copies are compared on type as well as value, because 8.0 == 8 and 1 == True
    in Python, and a bundle describing its own model two different ways is not one to
    predict with.
    """
    for label, given in (("model", stored), ("metadata model", declared)):
        if not isinstance(given, dict):
            raise ValueError(f"{label} parameters must be an object")
        missing = [name for name in MODEL_PARAMETERS if name not in given]
        if missing:
            raise ValueError(f"{label} parameters are missing {', '.join(missing)}")
        unexpected = sorted(set(given) - set(MODEL_PARAMETERS))
        if unexpected:
            raise ValueError(f"{label} parameters have unexpected {', '.join(unexpected)}")

    for name in MODEL_PARAMETERS:
        kept, said = stored[name], declared[name]
        if type(kept) is not type(said) or kept != said:
            raise ValueError(f"model parameter {name} is {kept!r}, metadata says {said!r}")
    return stored


def _tree_to_json(tree: DecisionTree) -> dict:
    return {"n_features": tree.n_features_, "root": _node_to_json(tree.root_)}


def _tree_from_json(data: dict, n_features: int) -> DecisionTree:
    tree = DecisionTree()
    tree.n_features_ = _json_int(data["n_features"], "tree n_features", minimum=1)
    if tree.n_features_ != n_features:
        raise ValueError(f"a tree takes {tree.n_features_} features, the forest takes {n_features}")
    tree.root_ = _node_from_json(data["root"], n_features)
    return tree


def _node_to_json(node: _Node) -> dict:
    saved = {"prediction": node.prediction, "probability": node.probability}
    if not node.is_leaf:
        saved["feature"] = node.feature
        saved["threshold"] = node.threshold
        saved["left"] = _node_to_json(node.left)
        saved["right"] = _node_to_json(node.right)
    return saved


def _node_from_json(data: dict, n_features: int) -> _Node:
    prediction = _json_int(data["prediction"], "leaf prediction")
    if prediction not in (0, 1):
        raise ValueError(f"leaf prediction must be 0 or 1, got {prediction}")
    probability = _json_number(data["probability"], "leaf probability")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"leaf probability must be between 0 and 1, got {probability}")
    node = _Node(prediction=prediction, probability=probability)

    present = [key for key in SPLIT_KEYS if key in data]
    if not present:
        return node
    if len(present) != len(SPLIT_KEYS):
        missing = [key for key in SPLIT_KEYS if key not in data]
        raise ValueError(f"split node is missing {', '.join(missing)}")

    feature = _json_int(data["feature"], "split feature")
    if not 0 <= feature < n_features:
        raise ValueError(f"split feature {feature} is outside the model's {n_features} features")

    node.feature = feature
    node.threshold = _json_number(data["threshold"], "split threshold")
    node.left = _node_from_json(data["left"], n_features)
    node.right = _node_from_json(data["right"], n_features)
    return node


def _state_to_json(state: TennisState) -> dict:
    """Player-keyed maps become sorted lists, since JSON object keys are strings."""
    return {
        "elo": _by_player(state.elo),
        "surface_elo": {surface: _by_player(v) for surface, v in sorted(state.surface_elo.items())},
        "matches_played": _by_player(state.matches_played),
        "elo_history": _by_player({k: list(v) for k, v in state.elo_history.items()}),
        "recent_results": _by_player({k: list(v) for k, v in state.recent_results.items()}),
        "serve_history": _by_player(
            {
                player: {stat: list(values) for stat, values in sorted(stats.items())}
                for player, stats in state.serve_history.items()
            }
        ),
        "h2h": _pairs(state.h2h),
        "surface_h2h": {surface: _pairs(v) for surface, v in sorted(state.surface_h2h.items())},
    }


def _state_from_json(data: dict) -> TennisState:
    return TennisState(
        elo=_player_map(data["elo"], "Elo", _json_number),
        surface_elo=_by_surface(
            data["surface_elo"],
            "surface Elo",
            lambda entries, label: _player_map(entries, label, _json_number),
        ),
        matches_played=_player_map(
            data["matches_played"], "matches played", lambda v, n: _json_int(v, n, minimum=0)
        ),
        elo_history=_player_map(
            data["elo_history"], "Elo history", lambda v, n: _history(v, n, _json_number)
        ),
        recent_results=_player_map(
            data["recent_results"],
            "recent results",
            lambda v, n: _history(v, n, lambda x, m: _json_int(x, m, minimum=0, maximum=1)),
        ),
        serve_history=_player_map(data["serve_history"], "serve history", _serve_history),
        h2h=_pair_map(data["h2h"], "head to head"),
        surface_h2h=_by_surface(data["surface_h2h"], "surface head to head", _pair_map),
    )


def _player_map(entries, label: str, parse) -> dict:
    """A [[player_id, value], ...] list back into a dict, refusing a repeated player."""
    if not isinstance(entries, list):
        raise ValueError(f"{label} must be a list")
    values = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"every {label} entry must be a player and a value")
        player_id = _json_int(entry[0], f"{label} player id", minimum=1)
        if player_id in values:
            raise ValueError(f"{label} lists player {player_id} twice")
        values[player_id] = parse(entry[1], f"{label} for player {player_id}")
    return values


def _pair_map(entries, label: str) -> dict:
    """A [[winner, loser, wins], ...] list back into a dict, refusing a repeated pair."""
    if not isinstance(entries, list):
        raise ValueError(f"{label} must be a list")
    wins = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 3:
            raise ValueError(f"every {label} entry must be a winner, a loser and a count")
        pair = (
            _json_int(entry[0], f"{label} winner id", minimum=1),
            _json_int(entry[1], f"{label} loser id", minimum=1),
        )
        if pair in wins:
            raise ValueError(f"{label} lists {pair[0]} against {pair[1]} twice")
        wins[pair] = _json_int(entry[2], f"{label} wins", minimum=0)
    return wins


def _by_surface(entries, label: str, parse) -> dict:
    if not isinstance(entries, dict):
        raise ValueError(f"{label} must be an object")
    unexpected = sorted(set(entries) - set(SURFACES))
    if unexpected:
        raise ValueError(f"{label} has unknown surfaces {', '.join(unexpected)}")
    return {surface: parse(values, f"{label} on {surface}") for surface, values in entries.items()}


def _history(values, label: str, parse) -> deque:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    if len(values) > MAX_HISTORY:
        raise ValueError(f"{label} holds {len(values)} entries, more than the {MAX_HISTORY} kept")
    return deque((parse(value, label) for value in values), maxlen=MAX_HISTORY)


def _serve_history(stats, label: str) -> dict:
    if not isinstance(stats, dict):
        raise ValueError(f"{label} must be an object")
    unexpected = sorted(set(stats) - set(SERVE_STATS))
    if unexpected:
        raise ValueError(f"{label} has unknown statistics {', '.join(unexpected)}")
    return {
        stat: _history(
            values,
            f"{label} {stat}",
            lambda value, name: _json_number(value, name, minimum=0.0, maximum=MAX_SERVE_RATE),
        )
        for stat, values in stats.items()
    }


def _profiles_to_json(profiles: dict[int, PlayerProfile]) -> list:
    return [
        {
            "player_id": profile.player_id,
            "atp_points": profile.snapshot.atp_points,
            "atp_rank": profile.snapshot.atp_rank,
            "age": profile.snapshot.age,
            "height": profile.snapshot.height,
            "last_seen_date": profile.last_seen_date,
            "name": profile.name,
        }
        for _, profile in sorted(profiles.items())
    ]


def _profiles_from_json(data) -> dict[int, PlayerProfile]:
    if not isinstance(data, list):
        raise ValueError("profiles must be a list")

    profiles: dict[int, PlayerProfile] = {}
    for saved in data:
        if not isinstance(saved, dict):
            raise ValueError("every profile must be an object")
        player_id = _json_int(saved["player_id"], "profile player id", minimum=1)
        if player_id in profiles:
            raise ValueError(f"profiles list player {player_id} twice")

        name = saved["name"]
        if name is not None and not isinstance(name, str):
            raise ValueError(f"name for player {player_id} must be text or absent, got {name!r}")

        snapshot = PlayerSnapshot(
            player_id=player_id,
            atp_points=_json_number(saved["atp_points"], f"atp_points for player {player_id}"),
            atp_rank=_json_number(saved["atp_rank"], f"atp_rank for player {player_id}"),
            age=_json_number(saved["age"], f"age for player {player_id}"),
            height=_json_number(saved["height"], f"height for player {player_id}"),
        )
        profiles[player_id] = PlayerProfile(
            snapshot=snapshot,
            last_seen_date=_json_int(
                saved["last_seen_date"], f"last_seen_date for player {player_id}"
            ),
            name=name,
        )
    return profiles


def _by_player(values: dict) -> list:
    return [[player, values[player]] for player in sorted(values)]


def _pairs(records: dict) -> list:
    return [[a, b, wins] for (a, b), wins in sorted(records.items())]
