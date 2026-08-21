"""Every image the compose files name is one CI actually builds.

The failure this exists for landed on the work server, not in CI::

    ✘ Image nikitaikonkin/ionograms-infer:latest  Error
      failed to resolve reference ...: not found
    Error response from daemon: failed to resolve reference ...

`docker-compose.hub.yml` gained an `infer` service; the CI matrix that pushes
to Docker Hub did not gain the matching entry. Nothing was wrong with either
file on its own, so nothing anywhere reported it -- and because `infer` has no
profile, a plain `up -d` tries to pull it and takes the *whole stack* down on
the failure rather than the one service. The first sign was a station that
would not come up.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: `nikitaikonkin/ionograms-infer:latest` -> `ionograms-infer`, whatever the
#: namespace and tag expand to.
IMAGE = re.compile(r"^\$\{IMAGE_NAMESPACE[^}]*\}/(?P<name>[A-Za-z0-9_.-]+):")


def compose(name: str) -> dict:
    return yaml.safe_load((DEPLOY / name).read_text())


def published() -> set[str]:
    """Image names the CI matrix builds and pushes."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    for job in workflow["jobs"].values():
        include = (job.get("strategy") or {}).get("matrix", {}).get("include")
        if include:
            return {entry["image"] for entry in include}
    raise AssertionError("no build matrix in the workflow")


def pulled(name: str) -> dict[str, str]:
    """Services in a compose file that pull a namespaced image, by service."""
    found = {}
    for service, body in compose(name)["services"].items():
        match = IMAGE.match(str(body.get("image", "")))
        if match:
            found[service] = match.group("name")
    return found


def test_the_hub_compose_pulls_only_images_ci_builds():
    missing = {s: i for s, i in pulled("docker-compose.hub.yml").items()
               if i not in published()}
    assert not missing, (
        f"docker-compose.hub.yml pulls {sorted(set(missing.values()))}, which "
        f"the CI matrix in .github/workflows/ci.yml does not build. `docker "
        f"compose up -d` on the work server will fail to resolve it and abort "
        f"the whole stack. Add it to the matrix beside its Dockerfile.")


def test_every_built_image_has_its_dockerfile():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    for job in workflow["jobs"].values():
        for entry in (job.get("strategy") or {}).get("matrix", {}).get("include", []):
            path = ROOT / entry["dockerfile"]
            assert path.exists(), f"{entry['image']}: no {entry['dockerfile']}"


def test_a_service_that_would_block_a_cold_start_has_no_profile_or_an_image():
    """A profiled service is only pulled when its profile is asked for, so it
    cannot break `up -d`. An unprofiled one must be pullable by everyone."""
    services = compose("docker-compose.hub.yml")["services"]
    for service, image in pulled("docker-compose.hub.yml").items():
        if not services[service].get("profiles"):
            assert image in published(), (
                f"{service} starts by default and pulls {image}, which is not "
                f"published. Either add it to the CI matrix or give the "
                f"service a profile so a cold start does not need it.")


def test_the_dev_compose_builds_rather_than_pulls():
    """It is the from-a-checkout file: a `build:` there cannot 404."""
    for service, body in compose("docker-compose.yml")["services"].items():
        assert "build" in body or "image" in body, service
        if "image" in body and not body.get("build"):
            assert not IMAGE.match(str(body["image"])), (
                f"{service} pulls a namespaced image from the dev compose; "
                f"that file exists so a checkout can run without the registry.")


# --------------------------------------------------------------------------
# Requirements that have to resolve on both architectures
# --------------------------------------------------------------------------
#
# The images are built for linux/amd64 and linux/arm64 from one requirements
# file, and pip only discovers a missing wheel halfway through a multi-minute
# cross-build. `tensorflow-cpu` has no aarch64 wheel and never has, so the
# arm64 leg failed with "from versions: none" -- which reads as a version-range
# problem rather than an architecture one, and cost a whole publish run.

PLATFORMS = ("x86_64", "aarch64")

REQUIREMENTS = sorted(DEPLOY.glob("requirements-*.txt"))

#: Distributions with no wheel for one of the architectures we build for, and
#: what to use instead. Checked by name, so a pin, a range or a marker on the
#: same distribution is all caught.
NO_WHEEL_FOR = {
    "tensorflow-cpu": ("aarch64", "tensorflow (the plain package is the CPU "
                                  "build on aarch64, same size)"),
    "tensorflow-aarch64": ("x86_64", "tensorflow-cpu"),
}


def requirements_of(path: Path) -> list:
    packaging = pytest.importorskip("packaging.requirements")
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        out.append(packaging.Requirement(line))
    return out


@pytest.mark.parametrize("path", REQUIREMENTS, ids=lambda p: p.name)
@pytest.mark.parametrize("arch", PLATFORMS)
def test_nothing_is_asked_for_on_an_architecture_that_has_no_wheel(path, arch):
    for req in requirements_of(path):
        bad_arch, instead = NO_WHEEL_FOR.get(req.name, (None, None))
        if bad_arch != arch:
            continue
        selected = req.marker is None or req.marker.evaluate(
            {"platform_machine": arch})
        assert not selected, (
            f"{path.name} installs {req.name} on {arch}, which publishes no "
            f"wheel for it -- pip fails the cross-build with 'from versions: "
            f"none'. Use {instead}, or exclude this architecture with a "
            f"`platform_machine` marker.")


@pytest.mark.parametrize("arch", PLATFORMS)
def test_exactly_one_tensorflow_is_selected_per_architecture(arch):
    """Two would conflict over the same import; none would fail at load."""
    chosen = [r.name for r in requirements_of(DEPLOY / "requirements-train.txt")
              if r.name.startswith("tensorflow")
              and (r.marker is None
                   or r.marker.evaluate({"platform_machine": arch}))]
    assert len(chosen) == 1, f"{arch}: selected {chosen or 'nothing'}"


def test_both_architectures_get_the_same_tensorflow_range():
    """One TensorFlow, two spellings. Different ranges either side would mean a
    model trained on one machine meeting a different runtime on the other."""
    ranges = {str(r.specifier)
              for r in requirements_of(DEPLOY / "requirements-train.txt")
              if r.name.startswith("tensorflow")}
    assert len(ranges) == 1, f"tensorflow pinned differently per arch: {ranges}"


def test_one_image_failing_does_not_cancel_the_others():
    """`fail-fast` defaults to true, and that is how a training-image problem
    stopped `ionograms-infer` reaching the registry."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    for name, job in workflow["jobs"].items():
        strategy = job.get("strategy")
        if strategy and "matrix" in strategy:
            assert strategy.get("fail-fast") is False, (
                f"job {name}: set `fail-fast: false`, or one image's failure "
                f"cancels the rest and a deploy blocks on an unrelated build.")
