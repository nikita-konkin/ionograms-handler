"""Loading a saved model, and proving it still behaves as it did.

These tests build their own artifacts rather than reaching for the research
project's: the point is the contract, and a fixture that has to be present on
this machine tests nothing on any other. One test does use a real file when it
is available, because "18 features named after a Brisbane column" is a fact
about the archive that no synthetic fixture would have caught.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.prediction import artifacts

joblib = pytest.importorskip("joblib")
sklearn_linear = pytest.importorskip("sklearn.linear_model")

#: A real legacy artifact, when this machine has one. Same arrangement as
#: `conftest.REAL_DATA`: the archive moves between drives.
REAL_ARTIFACT = Path(os.environ.get("MUF_TEST_MODEL", "/nonexistent/model.sav"))

FEATURES = [
    "X_lag_288", "X_trend_lag_288", "X_seasonal_lag_288", "X_residual_lag_288",
    "X_rolling_12_mean_lag_288", "X_rolling_12_std_lag_288",
    "X_rolling_12_min_lag_288", "X_rolling_12_max_lag_288",
    "hour", "minute",
]


@pytest.fixture
def saved(tmp_path):
    """A Ridge fitted on named columns, saved the way the archive saves them."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    target = frame.sum(axis=1)
    model = sklearn_linear.Ridge().fit(frame, target)
    path = tmp_path / "ridge_mae-0.1234_evals-0.sav"
    joblib.dump(model, path)
    return path


def test_a_saved_model_describes_its_own_inputs(saved):
    _, contract = artifacts.load(saved)
    assert contract.framework == "sklearn"
    assert contract.loader == "joblib"
    assert contract.capability == "slim"
    assert list(contract.features) == FEATURES
    assert contract.n_features == len(FEATURES)


def test_feature_order_is_preserved_not_sorted(saved):
    """Order is the contract. sklearn resolves inputs by position, not name."""
    _, contract = artifacts.load(saved)
    assert list(contract.features) != sorted(FEATURES)
    assert list(contract.features) == FEATURES


def test_the_golden_check_passes_on_an_unchanged_model(saved):
    estimator, contract = artifacts.load(saved)
    row = artifacts.golden_row(contract)
    value = artifacts.golden_check(estimator, contract, row, None)
    assert artifacts.golden_check(estimator, contract, row, value) == value


def test_the_golden_check_catches_a_changed_prediction(saved):
    """The defence against a library upgrade that changes behaviour silently.

    A version comparison cannot answer this: the versions differing is normal
    and usually harmless. What matters is whether the numbers moved.
    """
    estimator, contract = artifacts.load(saved)
    row = artifacts.golden_row(contract)
    value = artifacts.golden_check(estimator, contract, row, None)
    with pytest.raises(artifacts.ArtifactError, match="golden check failed"):
        artifacts.golden_check(estimator, contract, row, value + 1.0)


def test_skew_can_be_accepted_deliberately_and_is_recorded(saved):
    estimator, contract = artifacts.load(saved)
    row = artifacts.golden_row(contract)
    value = artifacts.golden_check(estimator, contract, row, None)

    with pytest.raises(artifacts.ArtifactError):
        artifacts.load_verified(saved, row, value + 1.0)

    _, _, quality = artifacts.load_verified(saved, row, value + 1.0, allow_skew=True)
    assert quality["golden"] == "failed"
    assert quality["version_skew"] is True


def test_a_replaced_file_is_not_served_from_cache(saved, tmp_path):
    """The models volume is writable by the training job, so a path is not an
    identity. Cache on mtime and size, never on the path alone."""
    _, first = artifacts.load(saved)
    assert first.n_features == len(FEATURES)

    rng = np.random.default_rng(1)
    fewer = FEATURES[:4]
    frame = pd.DataFrame(rng.normal(size=(50, len(fewer))), columns=fewer)
    replacement = sklearn_linear.Ridge().fit(frame, frame.sum(axis=1))
    os.utime(saved, (0, 0))
    joblib.dump(replacement, saved)

    _, second = artifacts.load(saved)
    assert second.n_features == len(fewer)


def test_an_unrecognised_file_says_what_it_saw(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not a model")
    with pytest.raises(artifacts.ArtifactError, match="not a format"):
        artifacts.load(path)


def test_a_keras2_saved_model_directory_names_the_remedy(tmp_path):
    """`nn_tuner.py` writes this format, and Keras 3 cannot read it."""
    directory = tmp_path / "keras_models" / "1_1"
    directory.mkdir(parents=True)
    (directory / "saved_model.pb").write_bytes(b"\x00")
    with pytest.raises(artifacts.ArtifactError, match="TF_USE_LEGACY_KERAS"):
        artifacts.sniff(directory)


def test_a_missing_artifact_is_reported_as_missing(tmp_path):
    with pytest.raises(artifacts.ArtifactError, match="not found"):
        artifacts.load(tmp_path / "absent.sav")


def test_sha256_changes_when_the_file_does(saved, tmp_path):
    before = artifacts.sha256(saved)
    saved.write_bytes(saved.read_bytes() + b"\x00")
    assert artifacts.sha256(saved) != before


@pytest.mark.skipif(not REAL_ARTIFACT.exists(),
                    reason="set MUF_TEST_MODEL to a real .sav to run this")
def test_the_real_archive_artifact_is_self_describing():
    """What the research project's files actually carry.

    Recorded as a test because it is the fact the whole legacy import rests
    on, and because the day it stops being true the importer must fail loudly
    rather than invent a feature order.
    """
    estimator, contract = artifacts.load(REAL_ARTIFACT)
    assert contract.framework == "sklearn"
    assert contract.n_features == len(contract.features)
    assert contract.features, "the artifact must name its own inputs"
    assert contract.env.get("sklearn"), "the fitting version must be recoverable"
    # Every predictor is a lagged transform of one aliased column, or a time
    # predictor. Nothing else is expected in the vertical format.
    from services.prediction import legacy_features
    recipe = legacy_features.parse(contract.features)
    assert recipe.lag > 0
