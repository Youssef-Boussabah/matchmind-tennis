import gzip
import hashlib
import json

import numpy as np
import pytest

from matchmind_v1.artifacts import (
    BUNDLE_VERSION,
    MODEL_PARAMETERS,
    load_bundle,
    save_bundle,
)
from matchmind_v1.tennis import FEATURE_NAMES, MatchContext, build_features


def saved(bundle, tmp_path, name="bundle.json.gz"):
    path = tmp_path / name
    save_bundle(bundle, path)
    return path


def payload_of(path):
    with gzip.open(path, "rb") as compressed:
        return json.loads(compressed.read().decode("utf-8"))


def test_a_bundle_round_trips(small_bundle, tmp_path):
    reloaded = load_bundle(saved(small_bundle, tmp_path))

    assert reloaded.metadata["bundle_version"] == BUNDLE_VERSION
    assert reloaded.metadata["history_end_date"] == 20241218
    assert reloaded.profiles.keys() == small_bundle.profiles.keys()


def test_the_model_predicts_identically_after_loading(small_bundle, tmp_path):
    reloaded = load_bundle(saved(small_bundle, tmp_path))
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, len(FEATURE_NAMES)))

    assert np.array_equal(
        small_bundle.model.predict_proba(X), reloaded.model.predict_proba(X)
    )
    assert np.array_equal(small_bundle.model.predict(X), reloaded.model.predict(X))
    assert len(reloaded.model.estimators) == len(small_bundle.model.estimators)
    assert reloaded.model.n_features_ == len(FEATURE_NAMES)


def test_the_state_round_trips(small_bundle, tmp_path):
    reloaded = load_bundle(saved(small_bundle, tmp_path))
    before, after = small_bundle.state, reloaded.state

    assert before.elo == after.elo
    assert before.surface_elo == after.surface_elo
    assert before.matches_played == after.matches_played
    assert before.h2h == after.h2h
    assert before.surface_h2h == after.surface_h2h
    for player_id in before.elo:
        assert list(before.elo_history_of(player_id)) == list(after.elo_history_of(player_id))
        assert list(before.recent_results_of(player_id)) == list(after.recent_results_of(player_id))
        for stat in ("ace", "first_serve_in", "break_points_saved"):
            assert list(before.serve_history_of(player_id, stat)) == list(
                after.serve_history_of(player_id, stat)
            )


def test_histories_come_back_capped(small_bundle, tmp_path):
    from matchmind_v1.tennis.state import MAX_HISTORY

    reloaded = load_bundle(saved(small_bundle, tmp_path))

    assert reloaded.state.elo_history[1].maxlen == MAX_HISTORY
    assert reloaded.state.serve_history[1]["ace"].maxlen == MAX_HISTORY


def test_features_are_the_same_against_the_loaded_state(small_bundle, tmp_path):
    reloaded = load_bundle(saved(small_bundle, tmp_path))
    context = MatchContext("Hard", 3, 128)

    for a, b in ((1, 2), (2, 3), (1, 3)):
        before = build_features(
            small_bundle.profiles[a].snapshot,
            small_bundle.profiles[b].snapshot,
            context,
            small_bundle.state,
        )
        after = build_features(
            reloaded.profiles[a].snapshot,
            reloaded.profiles[b].snapshot,
            context,
            reloaded.state,
        )
        assert before == after


def test_profiles_round_trip_exactly(small_bundle, tmp_path):
    reloaded = load_bundle(saved(small_bundle, tmp_path))

    assert reloaded.profiles == small_bundle.profiles
    assert reloaded.profiles[3].name is None
    assert reloaded.profiles[1].last_seen_date == 20241118


def test_predictions_survive_the_round_trip(small_bundle, tmp_path):
    reloaded = load_bundle(saved(small_bundle, tmp_path))

    before = small_bundle.predict(1, 2, surface="Clay")
    after = reloaded.predict(1, 2, surface="Clay")

    assert before.probability_a == after.probability_a


def test_saving_twice_gives_the_same_bytes(small_bundle, tmp_path):
    first = saved(small_bundle, tmp_path, "first.json.gz")
    second = saved(small_bundle, tmp_path, "second.json.gz")

    assert hashlib.sha256(first.read_bytes()).hexdigest() == (
        hashlib.sha256(second.read_bytes()).hexdigest()
    )


def test_the_gzip_header_carries_no_clock_or_filename(small_bundle, tmp_path):
    """Bytes 4-8 are the gzip mtime; bit 3 of the flags is a stored filename."""
    raw = saved(small_bundle, tmp_path).read_bytes()

    assert int.from_bytes(raw[4:8], "little") == 0
    assert raw[3] & 0b1000 == 0


def test_the_json_holds_no_nan_or_infinity(small_bundle, tmp_path):
    with gzip.open(saved(small_bundle, tmp_path), "rb") as compressed:
        text = compressed.read().decode("utf-8")

    for literal in ("NaN", "Infinity", "-Infinity"):
        assert literal not in text


def rewritten(path, change):
    payload = payload_of(path)
    change(payload)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with gzip.open(path, "wb") as compressed:
        compressed.write(text.encode("utf-8"))
    return path


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda p: p["metadata"].update(bundle_version=99), "not supported"),
        (lambda p: p["metadata"].pop("bundle_version"), "not supported"),
        (lambda p: p["metadata"].update(feature_names=["ELO_DIFF"]), "different feature set"),
        (lambda p: p["metadata"].pop("feature_names"), "different feature set"),
        (lambda p: p["metadata"].pop("history_end_date"), "history_end_date must be a whole"),
        (lambda p: p["model"].update(n_features=12), "takes 12 features"),
        (lambda p: p["model"]["trees"].pop(), "holds 2 trees"),
        (lambda p: p["profiles"].clear(), "no player profiles"),
    ],
)
def test_an_incompatible_bundle_is_refused(small_bundle, tmp_path, damage, message):
    path = rewritten(saved(small_bundle, tmp_path), damage)

    with pytest.raises(ValueError, match=message):
        load_bundle(path)


def test_metadata_survives_the_round_trip(small_bundle, tmp_path):
    reloaded = load_bundle(saved(small_bundle, tmp_path))

    assert reloaded.metadata["feature_names"] == list(FEATURE_NAMES)
    assert reloaded.metadata["dataset_rows"] == 18
    assert reloaded.metadata["model_parameters"]["n_estimators"] == 3


def corrupted(path, text):
    """Write raw text into the bundle's place, bypassing save_bundle entirely."""
    with gzip.open(path, "wb") as compressed:
        compressed.write(text.encode("utf-8"))
    return path


def every_leaf(node, change):
    change(node)
    for side in ("left", "right"):
        if side in node:
            every_leaf(node[side], change)


def set_leaves(payload, **values):
    """Overwrite a field on every node of the first tree."""
    every_leaf(payload["model"]["trees"][0]["root"], lambda node: node.update(**values))


def set_root(payload, **values):
    """Overwrite a field on the first tree's root split."""
    payload["model"]["trees"][0]["root"].update(**values)


def test_a_file_that_is_not_gzip_is_refused(small_bundle, tmp_path):
    path = tmp_path / "broken.json.gz"
    path.write_bytes(b"this is not a gzip stream")

    with pytest.raises(ValueError, match="not a readable prediction bundle"):
        load_bundle(path)


def test_a_truncated_bundle_is_refused(small_bundle, tmp_path):
    path = saved(small_bundle, tmp_path)
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])

    with pytest.raises(ValueError, match="not a readable prediction bundle"):
        load_bundle(path)


def test_malformed_json_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a readable prediction bundle"):
        load_bundle(corrupted(tmp_path / "bad.json.gz", '{"metadata": '))


def test_a_missing_section_is_refused(small_bundle, tmp_path):
    path = rewritten(saved(small_bundle, tmp_path), lambda p: p.pop("state"))

    with pytest.raises(ValueError, match="not a well-formed prediction bundle"):
        load_bundle(path)


def test_nan_probabilities_are_refused(small_bundle, tmp_path):
    """json.loads accepts the non-standard NaN literal, so the reader has to say no."""
    path = saved(small_bundle, tmp_path)
    payload = payload_of(path)
    for tree in payload["model"]["trees"]:
        every_leaf(tree["root"], lambda node: node.update(probability=float("nan")))
    text = json.dumps(payload)  # allow_nan defaults to True, writing bare NaN
    assert "NaN" in text

    with pytest.raises(ValueError, match="non-finite value NaN"):
        load_bundle(corrupted(path, text))


@pytest.mark.parametrize("literal", ["Infinity", "-Infinity"])
def test_infinities_are_refused(small_bundle, tmp_path, literal):
    path = saved(small_bundle, tmp_path)
    text = json.dumps(payload_of(path)).replace('"probability":0.5', f'"probability":{literal}', 1)
    text = text.replace('"threshold"', f'"x":{literal},"threshold"', 1)

    with pytest.raises(ValueError, match="non-finite value"):
        load_bundle(corrupted(path, text))


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda p: set_leaves(p, probability=1.5), "between 0 and 1"),
        (lambda p: set_leaves(p, probability=-0.1), "between 0 and 1"),
        (lambda p: set_leaves(p, probability="high"), "leaf probability must be a number"),
        (lambda p: set_leaves(p, prediction=7), "must be 0 or 1"),
        (lambda p: set_root(p, feature=999), "outside the model"),
        (lambda p: set_root(p, feature=-1), "outside the model"),
        (lambda p: set_root(p, feature="ELO_DIFF"), "must be a whole number"),
        (lambda p: set_root(p, threshold="low"), "threshold must be a number"),
        (lambda p: p["model"]["trees"][0]["root"].pop("right"), "missing right"),
        (lambda p: p["model"]["trees"][0].update(n_features=3), "a tree takes 3 features"),
        (lambda p: p["model"]["trees"][0].update(n_features=68), "a tree takes 68 features"),
        (lambda p: p["model"]["parameters"].update(n_estimators=99), "n_estimators is 99"),
        (lambda p: p["metadata"]["model_parameters"].update(max_depth=99), "metadata says 99"),
        (lambda p: p["metadata"].update(profile_count=99), "metadata says 99"),
    ],
)
def test_a_damaged_model_or_metadata_is_refused(small_bundle, tmp_path, damage, message):
    path = rewritten(saved(small_bundle, tmp_path), damage)

    with pytest.raises(ValueError, match=message):
        load_bundle(path)


def test_a_healthy_bundle_still_loads(small_bundle, tmp_path):
    """The guard rails must not reject the real thing."""
    reloaded = load_bundle(saved(small_bundle, tmp_path))

    assert len(reloaded.model.estimators) == 3
    assert reloaded.predict(1, 2).probability_a > 0.0


@pytest.mark.parametrize("literal", ["1e999", "-1e999"])
def test_numbers_too_large_for_a_float_are_refused(small_bundle, tmp_path, literal):
    """Valid JSON syntax, but Python parses it to infinity."""
    path = saved(small_bundle, tmp_path)
    text = json.dumps(payload_of(path)).replace('"probability":0.5', f'"probability":{literal}', 1)
    text = text.replace('"threshold":', f'"unused":{literal},"threshold":', 1)

    with pytest.raises(ValueError, match="out-of-range number"):
        load_bundle(corrupted(path, text))


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda p: p["model"].update(n_features=67.5), "model n_features must be a whole"),
        (lambda p: p["model"]["trees"][0].update(n_features=67.5), "tree n_features must be"),
        (lambda p: set_leaves(p, prediction=1.0), "leaf prediction must be a whole"),
        (lambda p: set_root(p, feature=2.0), "split feature must be a whole"),
        (lambda p: set_leaves(p, probability=True), "leaf probability must be a number"),
        (lambda p: p["model"]["parameters"].pop("max_depth"), "missing max_depth"),
        (lambda p: p["model"]["parameters"].update(criterion="gini"), "unexpected criterion"),
        (lambda p: p["metadata"].update(dataset_rows=0), "dataset_rows must be at least 1"),
        (lambda p: p["metadata"].update(dataset_rows=18.0), "dataset_rows must be a whole"),
        (lambda p: p["metadata"].update(package_version=""), "package_version must be a non-empty"),
        (lambda p: p["metadata"].update(dataset_sha256=None), "dataset_sha256 must be a non-empty"),
        (lambda p: p["metadata"].update(history_start_date=20250101), "after it ends"),
    ],
)
def test_loosely_typed_model_or_metadata_is_refused(small_bundle, tmp_path, damage, message):
    path = rewritten(saved(small_bundle, tmp_path), damage)

    with pytest.raises(ValueError, match=message):
        load_bundle(path)


def first_profile(payload, **values):
    payload["profiles"][0].update(**values)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda p: first_profile(p, player_id="1"), "profile player id must be a whole"),
        (lambda p: first_profile(p, player_id=1.0), "profile player id must be a whole"),
        (lambda p: first_profile(p, last_seen_date=20241118.0), "last_seen_date .* whole"),
        (lambda p: first_profile(p, atp_rank="3"), "atp_rank for player 1 must be a number"),
        (lambda p: first_profile(p, height=True), "height for player 1 must be a number"),
        (lambda p: first_profile(p, name=7), "must be text or absent"),
        (lambda p: p["profiles"].append(dict(p["profiles"][0])), "list player 1 twice"),
    ],
)
def test_a_loosely_typed_profile_is_refused(small_bundle, tmp_path, damage, message):
    path = rewritten(saved(small_bundle, tmp_path), damage)

    with pytest.raises(ValueError, match=message):
        load_bundle(path)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda p: p["state"]["elo"][0].__setitem__(1, "1e999"), "Elo for player 1 must be"),
        (lambda p: p["state"]["elo"][0].__setitem__(1, "1500"), "Elo for player 1 must be"),
        (lambda p: p["state"]["elo"].append(list(p["state"]["elo"][0])), "Elo lists player 1"),
        (lambda p: p["state"]["elo"][0].__setitem__(0, 0), "Elo player id must be at least 1"),
        (lambda p: p["state"]["matches_played"][0].__setitem__(1, -1), "must be at least 0"),
        (lambda p: p["state"]["matches_played"][0].__setitem__(1, 6.0), "must be a whole number"),
        (lambda p: p["state"]["elo_history"][0].__setitem__(1, [1.0] * 201), "more than the 200"),
        (lambda p: p["state"]["recent_results"][0].__setitem__(1, [2]), "must be at most 1"),
        (lambda p: p["state"]["recent_results"][0].__setitem__(1, [1.0]), "must be a whole number"),
        (lambda p: p["state"]["serve_history"][0][1]["ace"].append(140.0), "at most 100"),
        (lambda p: p["state"]["serve_history"][0][1].update(spin=[1.0]), "unknown statistics"),
        (lambda p: p["state"]["h2h"][0].__setitem__(2, -1), "wins must be at least 0"),
        (lambda p: p["state"]["h2h"].append(list(p["state"]["h2h"][0])), "lists 1 against 2"),
        (lambda p: p["state"]["surface_elo"].update(Ice=[]), "unknown surfaces Ice"),
    ],
)
def test_damaged_state_is_refused(small_bundle, tmp_path, damage, message):
    path = rewritten(saved(small_bundle, tmp_path), damage)

    with pytest.raises(ValueError, match=message):
        load_bundle(path)


def test_a_rated_player_without_a_profile_is_refused(small_bundle, tmp_path):
    def orphan(payload):
        payload["state"]["elo"].append([999, 1500.0])
        payload["state"]["elo"].sort()

    path = rewritten(saved(small_bundle, tmp_path), orphan)

    with pytest.raises(ValueError, match="rated players have no profile"):
        load_bundle(path)


def meta_params(payload, **values):
    """Change the metadata's copy of the model settings, leaving the stored copy alone."""
    payload["metadata"]["model_parameters"].update(values)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda p: p["metadata"]["model_parameters"].pop("max_depth"), "missing max_depth"),
        (
            lambda p: p["metadata"]["model_parameters"].pop("min_impurity_decrease"),
            "missing min_impurity_decrease",
        ),
        (lambda p: meta_params(p, criterion="gini"), "unexpected criterion"),
        (lambda p: p["metadata"].update(model_parameters={}), "metadata model .* are missing"),
    ],
)
def test_the_metadata_must_name_every_model_parameter(small_bundle, tmp_path, damage, message):
    path = rewritten(saved(small_bundle, tmp_path), damage)

    with pytest.raises(ValueError, match=message):
        load_bundle(path)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"max_depth": 3.0}, "max_depth is 3, metadata says 3.0"),
        ({"n_estimators": 3.0}, "n_estimators is 3, metadata says 3.0"),
        ({"random_state": 0.0}, "random_state is 0, metadata says 0.0"),
        ({"min_samples_split": 2.0}, "min_samples_split is 2, metadata says 2.0"),
        ({"bootstrap": 1}, "bootstrap is True, metadata says 1"),
        ({"min_impurity_decrease": 0}, "min_impurity_decrease is 0.0, metadata says 0"),
    ],
)
def test_a_parameter_of_the_wrong_type_is_refused(small_bundle, tmp_path, values, message):
    """8.0 == 8 and 1 == True in Python, so the two copies are compared on type too."""
    path = rewritten(saved(small_bundle, tmp_path), lambda p: meta_params(p, **values))

    with pytest.raises(ValueError, match=message):
        load_bundle(path)


def test_matching_parameter_copies_still_load(small_bundle, tmp_path):
    reloaded = load_bundle(saved(small_bundle, tmp_path))

    assert set(reloaded.metadata["model_parameters"]) == set(MODEL_PARAMETERS)
    assert reloaded.metadata["model_parameters"]["min_impurity_decrease"] == 0.0
