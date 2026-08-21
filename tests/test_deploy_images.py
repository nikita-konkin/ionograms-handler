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
