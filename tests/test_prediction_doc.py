"""``docs/prediction.md`` against the code it describes.

The document exists because the pipeline's rules are the kind that fail
silently when broken -- feed a model the wrong column order and it returns a
plausible megahertz rather than an error -- so writing them down is worth
doing. But a reference that can drift is worse than none: `architecture.md`
sec. 4.4 sat marked **[blocked]** for weeks after the service was built and
running, and the marker was believed.

So the mechanical half of the document is re-derived from the source here.
Constants, the baseline names, the frameworks the slim image can load, and the
two schema constraints that make promotion a refusal rather than a warning.

The *narrative* half is not guarded and cannot be: "truth is measured, never
tracked" is a claim about intent that a word-search would only ever confirm
tautologically. The docstrings in `services/prediction/` are where that lives,
and they sit beside the code that has to honour it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "prediction.md"
SCHEMA = Path(__file__).resolve().parents[1] / "services" / "api" / "schema.sql"


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_the_document_is_there_at_all(doc):
    # Guards against every assertion below passing over an empty read.
    assert "## 1. Overview" in doc


def test_the_grid_and_the_minimum_window_match_dataset(doc):
    from services.prediction import dataset

    assert dataset.DEFAULT_STEP_S == 300
    assert f"regular {dataset.DEFAULT_STEP_S} s grid" in doc
    # Quoted as the expression, because the *reason* is that it is two whole
    # decomposition periods -- not that it happens to equal 576.
    assert "`MIN_SAMPLES = 2 × 288`" in doc
    assert dataset.MIN_SAMPLES == 2 * 288


def test_the_decomposition_period_matches_legacy_features(doc):
    from services.prediction import legacy_features

    assert legacy_features.DEFAULT_DECOMPOSITION_PERIOD == 288
    assert "288 —\none day at five-minute sampling — is the default" in doc


def test_the_golden_tolerance_matches_artifacts(doc):
    from services.prediction import artifacts

    assert artifacts.GOLDEN_TOLERANCE == 1e-6
    assert "`GOLDEN_TOLERANCE = 1e-6`" in doc


def test_the_slim_frameworks_match_artifacts(doc):
    from services.prediction import artifacts

    assert artifacts.SLIM_FRAMEWORKS == ("sklearn", "xgboost")
    assert "`slim` (sklearn, xgboost)" in doc


def test_the_baselines_and_horizons_match_scoring(doc):
    from services.prediction import scoring

    quoted = re.search(r"`BASELINES = \(([^)]*)\)`", doc)
    assert quoted, "the document no longer quotes BASELINES"
    named = tuple(re.findall(r'"([^"]+)"', quoted.group(1)))
    assert named == scoring.BASELINES

    horizons = re.search(r"`HORIZONS = \(([^)]*)\)`", doc)
    assert horizons, "the document no longer quotes HORIZONS"
    assert tuple(int(n) for n in re.findall(r"\d+", horizons.group(1))) \
        == scoring.HORIZONS


def test_the_promotion_constraints_are_quoted_verbatim(doc):
    """The two CHECKs are the whole reason promotion is a refusal.

    Quoted rather than paraphrased, so a schema change that loosens either one
    has to come past this test on its way to making the document wrong.
    """
    schema = SCHEMA.read_text(encoding="utf-8")
    for constraint in (
        "CHECK (active = 0 OR target_src = 'measured')",
        "CHECK (active = 0 OR (tx IS NOT NULL AND rx IS NOT NULL))",
    ):
        assert constraint in schema, f"gone from the schema: {constraint}"
        assert constraint in doc, f"not quoted in the document: {constraint}"


def test_the_three_origins_match_the_importer(doc):
    """`--origin` accepts exactly the three the document tabulates."""
    source = (Path(__file__).resolve().parents[1]
              / "services" / "prediction" / "importer.py").read_text()
    choices = re.search(r'"--origin".*?choices=\(([^)]*)\)', source, re.S)
    assert choices, "importer no longer declares --origin choices"
    for origin in re.findall(r'"([^"]+)"', choices.group(1)):
        assert f"| `{origin}` |" in doc, f"origin not in the document: {origin}"


def test_the_read_surfaces_it_names_exist(doc):
    """Every route the runbook tells an operator to open."""
    routes = {
        "/forecast": "read_routes.py",
        "/ui/series": "web_routes.py",
        "/ui/forecast": "web_routes.py",
    }
    api = Path(__file__).resolve().parents[1] / "services" / "api"
    for route, filename in routes.items():
        source = (api / filename).read_text(encoding="utf-8")
        assert f'@router.get("{route}")' in source, f"route gone: {route}"
        assert f"`{route}`" in doc, f"route not in the document: {route}"


def test_the_named_model_query_parameters_are_real(doc):
    """`?model=` and `?forecast=` are what make a comparison visible at all.

    Both were added because the default is active-only; a document that names
    the wrong parameter sends an operator to a page that correctly draws
    nothing, which is indistinguishable from a broken deployment.
    """
    api = Path(__file__).resolve().parents[1] / "services" / "api"
    assert "model: int | None = None" in (api / "read_routes.py").read_text()
    assert "forecast: int | None = None" in (api / "web_routes.py").read_text()
    assert "`?model=<id>`" in doc
    assert "`?forecast=<id>`" in doc
