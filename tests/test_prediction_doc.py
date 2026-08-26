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


# --------------------------------------------------------------------------
# The console path
#
# Added when uploading and training moved into the browser. Same reasoning as
# everything above: these are the parts of the document a code change can
# falsify without anyone reading it again.
# --------------------------------------------------------------------------

ARCHITECTURE = Path(__file__).resolve().parents[1] / "docs" / "architecture.md"


@pytest.fixture(scope="module")
def architecture() -> str:
    return ARCHITECTURE.read_text(encoding="utf-8")


def test_the_store_layout_is_quoted_as_the_code_builds_it(doc):
    """The layout is the one fact a reader might copy into a shell command."""
    from services.prediction import store

    assert store.OBJECTS == "objects"
    assert "/models/objects/<first two hex>/<the full 64-hex sha256>" in doc
    # And the mode, which is the reason a `chmod` in the runbook would fail.
    assert "0444" in doc


def test_the_estimators_the_document_lists_are_the_ones_that_exist(doc):
    from services.prediction import train

    assert train.DEFAULT_ESTIMATOR == "huber"

    # Read out of the sentence rather than compared against a copy of the
    # tuple: the list has grown once already, and a guard that has to be
    # edited in step with the thing it guards stops being a guard.
    sentence = re.search(r"Estimators are ([^.]+)\.", doc)
    assert sentence, "the document no longer lists the estimators"
    listed = set(re.findall(r"`(\w+)`", sentence.group(1)))
    assert listed == set(train.ESTIMATORS), (
        f"the document lists {sorted(listed)}; the module offers "
        f"{sorted(train.ESTIMATORS)}")
    assert f"`{train.DEFAULT_ESTIMATOR}` (default)" in sentence.group(1)


def test_the_committee_members_are_documented_with_their_weighting(doc):
    """The port's one deliberate divergence from `muf`, and its caveat.

    Both are the kind of thing a reader has to be able to find six months
    later: why the voting weights are not `muf`'s, and why the stack's inner
    folds are not chronological. Losing either from the document leaves a
    reimplementation that looks like a faithful port and is not.
    """
    from services.prediction import train

    for member in train.MEMBERS:
        assert f"`{member}`" in doc, f"committee member undocumented: {member}"
    for kind in train.ENSEMBLES:
        assert f"`{kind}`" in doc

    assert "inverse MAE" in doc, "the voting weights' basis is not stated"
    assert "cross_val_predict" in doc, "the stacking caveat is not stated"
    assert "cv_chronological" in doc


def test_the_console_routes_exist_and_are_named(doc, architecture):
    """Every route the console panels post to, and the pull.

    Checked against `architecture.md` as well: sec. 5.3 tabulates them, and a
    table of routes is exactly the kind of thing that goes stale silently.
    """
    api = Path(__file__).resolve().parents[1] / "services" / "api"
    control = (api / "control_routes.py").read_text(encoding="utf-8")
    read = (api / "read_routes.py").read_text(encoding="utf-8")

    for route in ("/models/upload", "/models/train"):
        assert f'@router.post("{route}")' in control, f"route gone: {route}"
        assert f"`POST {route}`" in architecture, f"not tabulated: {route}"

    for route in ("/models/uploads", "/models/jobs"):
        assert f'@router.get("{route}")' in read, f"route gone: {route}"
        assert route in architecture

    assert '@router.get("/models/{model_id}/artifact")' in read
    assert "`GET /models/<id>/artifact`" in architecture
    assert "`GET /models/<id>/artifact`" in doc


def test_the_worker_poll_intervals_match_what_the_document_promises(doc,
                                                                    architecture):
    """"Settles in about ten seconds" is a claim the default can falsify."""
    from services.prediction import registrar, trainer

    assert registrar.DEFAULT_INTERVAL_S == 10
    assert trainer.DEFAULT_INTERVAL_S == 60
    assert "settles in about ten seconds" in doc
    assert "`registrar` (10 s), `trainer` (60 s)" in architecture


def test_the_documents_agree_that_dvc_was_considered_and_declined(doc,
                                                                  architecture):
    """A "why not" is worth guarding only against being quietly dropped.

    The layout choice is otherwise unexplainable -- a two-character fan-out
    directory reads as arbitrary unless the document says whose convention it
    is and why the tool that owns it is not here.
    """
    assert "DVC's cache layout" in doc
    assert "DVC itself is *not* used" in doc
    assert "DVC is not used" in architecture
    assert not (ARCHITECTURE.parent.parent / ".dvc").exists(), \
        "DVC is in the tree now; both documents say it deliberately is not"


def test_only_the_two_workers_may_write_the_store(doc):
    """`models` is read-write in `registrar` and `trainer` and nowhere else.

    The compose files are the enforcement; this is the check that nobody
    loosened one of them in passing. `infer` runs code out of these files and
    must not be able to replace one -- `Dockerfile.infer` says so at length.
    """
    compose = Path(__file__).resolve().parents[1] / "deploy"
    for name in ("docker-compose.yml", "docker-compose.hub.yml"):
        text = (compose / name).read_text(encoding="utf-8")
        writable = re.findall(r"^      - models:/models$", text, re.M)
        readonly = re.findall(r"^      - models:/models:ro$", text, re.M)
        assert len(writable) == 2, \
            f"{name}: {len(writable)} services can write the object store, expected 2"
        assert readonly, f"{name}: nothing mounts the store read-only any more"


def test_the_poll_slice_is_quoted_as_the_code_sets_it(doc, architecture):
    """"Within about ten seconds" is a promise `POLL_S` can falsify."""
    from services.prediction import infer

    assert infer.POLL_S == 10
    assert infer.DEFAULT_INTERVAL_S == 21600
    assert "`POLL_S = 10` second slices" in doc
    assert "cut into 10 s slices" in architecture


def test_the_run_route_exists_and_both_documents_name_it(doc, architecture):
    api = Path(__file__).resolve().parents[1] / "services" / "api"
    assert '@router.post("/models/run")' in (api / "control_routes.py").read_text()
    assert '@router.get("/models/runs")' in (api / "read_routes.py").read_text()
    assert "`POST /models/run`" in architecture
    assert "`GET /models/runs`" in architecture
    assert "infer_job" in doc


def test_a_requested_pass_that_writes_nothing_is_a_failure(doc):
    """The one place the on-demand path deliberately differs from the loop.

    Documented because it looks like an inconsistency until the reason is
    stated, and an inconsistency nobody explained is one somebody "fixes".
    """
    source = (Path(__file__).resolve().parents[1] / "services" / "prediction"
              / "infer.py").read_text(encoding="utf-8")
    assert "state = queues.DONE if written else queues.FAILED" in source
    assert "is `failed`, not `done`" in doc
