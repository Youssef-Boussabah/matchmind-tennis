import copy

import pytest

from conftest import sample_profile
from matchmind_v1.prediction import PredictionBundle, normalise_name
from matchmind_v1.tennis import MatchContext, PlayerSnapshot


def rebuilt(bundle, profiles):
    return PredictionBundle(
        model=bundle.model, state=bundle.state, profiles=profiles, metadata=bundle.metadata
    )


@pytest.mark.parametrize(
    "query", ["Alpha Player", "alpha player", "  Alpha   Player  ", "ALPHA PLAYER"]
)
def test_a_name_resolves_whatever_its_case_and_spacing(small_bundle, query):
    assert small_bundle.resolve(query).player_id == 1


@pytest.mark.parametrize("query", [2, "2", " 2 "])
def test_a_numeric_id_resolves(small_bundle, query):
    assert small_bundle.resolve(query).player_id == 2


def test_a_player_with_no_name_is_still_reachable_by_id(small_bundle):
    profile = small_bundle.resolve(3)

    assert profile.name is None
    assert profile.label == "3"


def test_an_unknown_name_is_refused(small_bundle):
    with pytest.raises(ValueError, match="no player named"):
        small_bundle.resolve("Nobody Here")


def test_an_unknown_id_is_refused(small_bundle):
    with pytest.raises(ValueError, match="no player with id 999"):
        small_bundle.resolve(999)


def test_an_ambiguous_name_is_refused_with_the_candidates(small_bundle):
    profiles = dict(small_bundle.profiles) | {4: sample_profile(4, "alpha  PLAYER")}

    with pytest.raises(ValueError, match=r"matches several players \(1, 4\)"):
        rebuilt(small_bundle, profiles).resolve("Alpha Player")


def test_a_numeric_name_is_read_as_an_id(small_bundle):
    profiles = dict(small_bundle.profiles) | {7: sample_profile(7, "2")}
    bundle = rebuilt(small_bundle, profiles)

    assert bundle.resolve("2").player_id == 2
    assert bundle.resolve(7).name == "2"


def test_normalise_name_collapses_whitespace_and_case():
    assert normalise_name("  Jannik   SINNER ") == normalise_name("jannik sinner")


def test_prediction_probabilities_are_complements(small_bundle):
    prediction = small_bundle.predict("Alpha Player", "Beta Player")

    assert prediction.probability_a + prediction.probability_b == 1.0
    assert 0.0 <= prediction.probability_a <= 1.0


def test_reversing_the_query_swaps_exactly(small_bundle):
    forward = small_bundle.predict("Alpha Player", "Beta Player", surface="Clay", best_of=5)
    reverse = small_bundle.predict("Beta Player", "Alpha Player", surface="Clay", best_of=5)

    assert forward.probability_a == reverse.probability_b
    assert forward.probability_b == reverse.probability_a
    assert forward.winner.player_id == reverse.winner.player_id


def test_the_internal_order_does_not_depend_on_the_query_order(small_bundle):
    """Whichever way round the ids are given, one feature vector is scored once."""
    for a, b in ((1, 3), (3, 1), (2, 3), (3, 2)):
        forward = small_bundle.predict(a, b)
        reverse = small_bundle.predict(b, a)
        assert forward.probability_a == reverse.probability_b


def test_a_player_cannot_play_themselves(small_bundle):
    with pytest.raises(ValueError, match="cannot play themselves"):
        small_bundle.predict("Alpha Player", 1)


def test_prediction_reports_the_match_it_was_asked_about(small_bundle):
    prediction = small_bundle.predict(1, 2, surface="grass", best_of=5, draw_size=32)

    assert (prediction.surface, prediction.best_of, prediction.draw_size) == ("Grass", 5, 32)
    assert prediction.history_end_date == 20241218
    assert prediction.player_a.label == "Alpha Player"
    assert prediction.player_b.last_seen_date == 20241118


def test_predicting_leaves_the_bundle_alone(small_bundle):
    before = copy.deepcopy(small_bundle.state)
    profiles_before = dict(small_bundle.profiles)

    small_bundle.predict(1, 2)
    small_bundle.predict(2, 3, surface="Clay")

    assert small_bundle.state.elo == before.elo
    assert small_bundle.state.h2h == before.h2h
    assert small_bundle.state.surface_elo == before.surface_elo
    assert small_bundle.state.matches_played == before.matches_played
    for player_id in before.elo:
        assert list(small_bundle.state.elo_history_of(player_id)) == list(
            before.elo_history_of(player_id)
        )
    assert small_bundle.profiles == profiles_before


@pytest.mark.parametrize("surface", ["Ice", "hardcourt", "", "Indoor"])
def test_an_unknown_surface_is_refused(small_bundle, surface):
    with pytest.raises(ValueError, match="surface must be one of"):
        small_bundle.predict(1, 2, surface=surface)


@pytest.mark.parametrize("best_of", [1, 4, 7, True, 3.0, "3"])
def test_best_of_must_be_three_or_five(small_bundle, best_of):
    with pytest.raises(ValueError, match="best_of must be 3 or 5"):
        small_bundle.predict(1, 2, best_of=best_of)


@pytest.mark.parametrize("draw_size", [1, 0, -8, 32.5, True, "32"])
def test_draw_size_must_be_a_sensible_integer(small_bundle, draw_size):
    with pytest.raises(ValueError, match="draw_size must be an integer"):
        small_bundle.predict(1, 2, draw_size=draw_size)


@pytest.mark.parametrize("draw_size", [2, 9, 12, 28, 128])
def test_draw_sizes_the_source_actually_uses_are_accepted(small_bundle, draw_size):
    assert small_bundle.predict(1, 2, draw_size=draw_size).draw_size == draw_size


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("player_id", 0, "player_id must be a positive integer"),
        ("player_id", -3, "player_id must be a positive integer"),
        ("player_id", 1.0, "player_id must be a positive integer"),
        ("player_id", True, "player_id must be a positive integer"),
        ("age", 0.0, "age must be positive"),
        ("age", -1.0, "age must be positive"),
        ("age", float("nan"), "age must be finite"),
        ("atp_rank", 0.0, "atp_rank must be at least 1"),
        ("atp_rank", float("inf"), "atp_rank must be finite"),
        ("atp_points", -1.0, "atp_points cannot be negative"),
        ("height", 71.0, "height must be in centimetres"),
        ("height", "tall", "height must be a number"),
    ],
)
def test_snapshot_rejects_impossible_attributes(field, value, message):
    fields = {
        "player_id": 1,
        "atp_points": 2000.0,
        "atp_rank": 5.0,
        "age": 25.0,
        "height": 185.0,
    }
    with pytest.raises(ValueError, match=message):
        PlayerSnapshot(**(fields | {field: value}))


def test_a_plausible_snapshot_and_context_are_accepted():
    PlayerSnapshot(player_id=206173, atp_points=1.0, atp_rank=1.0, age=15.4, height=100.0)
    assert MatchContext("Carpet", 5, 2).surface == "Carpet"
