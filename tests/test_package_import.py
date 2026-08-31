import tomllib
from pathlib import Path

import matchmind_v1
from matchmind_v1 import trees
from matchmind_v1.artifacts import BUNDLE_VERSION
from matchmind_v1.data import DATASET_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN
from matchmind_v1.prediction import PredictionBundle
from matchmind_v1.tennis import FEATURE_NAMES
from matchmind_v1.trees import DecisionTree, RandomForest


def test_the_package_and_project_versions_agree():
    """One version, in two files that have no way to check each other at runtime."""
    pyproject = tomllib.loads(Path(__file__).parents[1].joinpath("pyproject.toml").read_text())

    assert matchmind_v1.__version__ == pyproject["project"]["version"] == "1.0.0"


def test_public_api():
    assert trees.__all__ == ["DecisionTree", "RandomForest"]
    assert isinstance(DecisionTree, type)
    assert isinstance(RandomForest, type)


def test_dataset_schema_is_metadata_features_and_one_target():
    assert len(FEATURE_NAMES) == 67
    assert len(DATASET_COLUMNS) == len(METADATA_COLUMNS) + len(FEATURE_NAMES) + 1
    assert DATASET_COLUMNS[-1] == TARGET_COLUMN


def test_prediction_entry_points_are_importable():
    assert isinstance(BUNDLE_VERSION, int)
    assert hasattr(PredictionBundle, "predict")
    assert hasattr(PredictionBundle, "resolve")
