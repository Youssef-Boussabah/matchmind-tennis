import numpy as np
import pandas as pd
import pytest

# One plausible raw ATP match row. Tests override the few fields they care about.
DEFAULT_MATCH = {
    "tourney_id": "2020-100",
    "tourney_name": "Test Open",
    "surface": "Hard",
    "draw_size": 32,
    "tourney_date": 20200106,
    "match_num": 1,
    "round": "R32",
    "best_of": 3,
    "winner_id": 1,
    "winner_ht": 185.0,
    "winner_age": 25.0,
    "winner_rank": 10.0,
    "winner_rank_points": 2000.0,
    "loser_id": 2,
    "loser_ht": 190.0,
    "loser_age": 28.0,
    "loser_rank": 40.0,
    "loser_rank_points": 900.0,
    "w_ace": 10,
    "w_df": 2,
    "w_svpt": 100,
    "w_1stIn": 60,
    "w_1stWon": 45,
    "w_2ndWon": 20,
    "w_bpSaved": 3,
    "w_bpFaced": 5,
    "l_ace": 4,
    "l_df": 5,
    "l_svpt": 95,
    "l_1stIn": 55,
    "l_1stWon": 35,
    "l_2ndWon": 18,
    "l_bpSaved": 2,
    "l_bpFaced": 6,
}


@pytest.fixture
def raw_matches():
    """Builds a raw match frame from one override dict per row."""

    def make(*rows):
        frame = pd.DataFrame([DEFAULT_MATCH | row for row in rows])
        frame["source_year"] = frame["tourney_date"] // 10000
        frame["source_row"] = np.arange(len(frame))
        return frame

    return make


@pytest.fixture
def separable():
    """40 samples, 2 features, separable on feature 0 at 4.5."""
    feature_0 = np.arange(40) % 10
    feature_1 = (np.arange(40) * 7) % 13
    X = np.column_stack([feature_0, feature_1]).astype(float)
    y = (feature_0 >= 5).astype(int)
    return X, y


@pytest.fixture
def xor_like():
    """A problem no single split can solve, so trees must go deeper."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 2)).astype(float)
    y = (X[:, 0].astype(int) ^ X[:, 1].astype(int)).astype(int)
    return X, y


@pytest.fixture
def noisy_blobs():
    """150 samples, 4 features, deterministic and not linearly separable."""
    rng = np.random.default_rng(7)
    n = 150
    X = rng.normal(size=(n, 4))
    logit = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2]
    y = (logit + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def sample_profile(player_id, name=None, *, rank=10.0, points=2000.0, age=25.0, height=185.0):
    from matchmind_v1.prediction import PlayerProfile
    from matchmind_v1.tennis import PlayerSnapshot

    return PlayerProfile(
        snapshot=PlayerSnapshot(
            player_id=player_id, atp_points=points, atp_rank=rank, age=age, height=height
        ),
        last_seen_date=20241118,
        name=name,
    )


@pytest.fixture
def small_bundle():
    """A complete but tiny bundle: a real forest, a replayed state and three players."""
    import matchmind_v1
    from matchmind_v1.artifacts import MODEL_PARAMETERS
    from matchmind_v1.prediction import PredictionBundle
    from matchmind_v1.tennis import (
        FEATURE_NAMES,
        CompletedMatch,
        PlayerMatchStats,
        TennisState,
    )
    from matchmind_v1.trees import RandomForest

    stats = PlayerMatchStats(
        ace=10,
        double_fault=2,
        service_points=100,
        first_serve_in=60,
        first_serve_won=45,
        second_serve_won=20,
        break_points_saved=3,
        break_points_faced=5,
    )
    state = TennisState()
    for round_number in range(6):
        for p1, p2 in ((1, 2), (2, 3), (1, 3)):
            state.update(
                CompletedMatch(
                    player1_id=p1,
                    player2_id=p2,
                    surface="Hard" if round_number % 2 else "Clay",
                    result=round_number % 2,
                    player1_stats=stats,
                    player2_stats=stats,
                )
            )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURE_NAMES)))
    y = (X[:, 6] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    model = RandomForest(n_estimators=3, max_depth=3, random_state=0).fit(X, y)

    profiles = {
        1: sample_profile(1, "Alpha Player", rank=3.0),
        2: sample_profile(2, "Beta Player", rank=40.0, age=29.0),
        3: sample_profile(3, None, rank=120.0, age=22.0),
    }
    return PredictionBundle(
        model=model,
        state=state,
        profiles=profiles,
        metadata={
            "package_version": matchmind_v1.__version__,
            "dataset_rows": 18,
            "dataset_sha256": "0" * 64,
            "feature_names": list(FEATURE_NAMES),
            "model_parameters": {name: getattr(model, name) for name in MODEL_PARAMETERS},
            "history_start_date": 19910107,
            "history_end_date": 20241218,
            "profile_count": 3,
        },
    )
