"""Keep routine feature updates from launching the full compatibility matrix."""
from pathlib import Path

import yaml


WORKFLOW = yaml.load(
    (Path(__file__).resolve().parents[1] / ".github/workflows/compatibility.yml").read_text(),
    Loader=yaml.BaseLoader,  # Keep GitHub's `on` key instead of YAML 1.1 boolean coercion.
)


def test_automatic_events_are_limited_to_integration_branches():
    events = WORKFLOW["on"]
    assert set(events) == {"push", "pull_request", "workflow_dispatch"}
    assert events["push"]["branches"] == ["dev", "main"]
    assert events["pull_request"]["branches"] == ["dev", "main"]
    option = events["workflow_dispatch"]["inputs"]["full_matrix"]
    assert option["type"] == "boolean"
    assert option["default"] == "false"


def test_matrix_requires_validation_and_does_not_run_for_pr_updates():
    matrix = WORKFLOW["jobs"]["tests"]
    assert matrix["needs"] == "data-resolution"
    assert matrix["if"] == ("github.event_name == 'push' || "
                            "(github.event_name == 'workflow_dispatch' && inputs.full_matrix)")
    assert matrix["strategy"]["fail-fast"] == "true"
    assert matrix["strategy"]["max-parallel"] == "3"
    # Keep the full intended compatibility coverage for integration branches.
    assert len(matrix["strategy"]["matrix"]["os"]) == 3
    assert len(matrix["strategy"]["matrix"]["python"]) == 3
    assert len(matrix["strategy"]["matrix"]["opencv"]) == 2


def test_shared_test_requirements_are_installed_by_both_jobs():
    for job in WORKFLOW["jobs"].values():
        commands = [step.get("run", "") for step in job["steps"]]
        assert any("pip install -r requirements-test.txt" in cmd for cmd in commands)
    assert WORKFLOW["concurrency"]["cancel-in-progress"] == "true"
    assert "github.event.pull_request.number || github.ref" in WORKFLOW["concurrency"]["group"]
