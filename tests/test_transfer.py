import asyncio
import pytest

from ganglion.arena.harness import experiment
from ganglion.arena.manipulation import ManipulationWorld, CONDITION_RGB
from ganglion.arena.target import TARGET_RGB
from ganglion.evaluation.transfer import Profile, exercise


def profile():
    return {"name": "test transfer", "application": "synthetic", "expected_session": 0,
        "client_size": [640, 360], "watches": {
            "source": {"region": [70, 130, 60, 50], "color_rgb": list(TARGET_RGB)},
            "condition": {"region": [40, 280, 560, 25], "color_rgb": list(CONDITION_RGB)}},
        "cases": [{"name": "move", "description": "independent fixture acceptance",
            "source": "source", "destination": [520, 180], "condition": "condition",
            "before": {"source": True, "condition": False}, "after": {"source": False, "condition": True}}]}


@pytest.mark.parametrize("fault", ["wrong_session", "changed_board", "none"])
def test_transfer_uses_mcp_and_checks_before_injecting(tmp_path, fault):
    data = profile()
    if fault == "wrong_session":
        data["expected_session"] = 99
    elif fault == "changed_board":
        data["watches"]["source"]["color_rgb"] = [255, 0, 255]
    taught = Profile.model_validate(data)
    with experiment("synthetic", ManipulationWorld, scenario="manipulation") as host:
        host.reset({"id": "transfer", "mode": "drop", "seed": 1})
        report = asyncio.run(exercise(host.endpoint, taught, tmp_path))
        truth = host.finish()
    downs = [e for e in truth if e["kind"] == "pointer_down"]
    if fault == "none":
        assert report["passed"] and len(downs) == 1
        assert any(e["kind"] == "drop_accepted" for e in truth)
        assert report["cases"][0]["post_observation_after_release"]
        assert (tmp_path / "move-after.jpg").exists()
    else:
        assert not report["passed"] and not downs
        assert not any(e["kind"] == "intent_started" for e in report["events"])
        if fault == "wrong_session":
            assert "different Windows session" in report["error"]
            assert not any(e["kind"] == "lease_claimed" for e in report["events"])
        else:
            assert "precondition_failed" in report["cases"][0]["error"]
    if fault != "wrong_session":
        assert report["final_state"] == "halted" and not report["pending_commands"]


@pytest.mark.parametrize("fault", ["missing_watch", "out_of_bounds", "satisfied_condition", "unsafe_filename"])
def test_bad_profiles_cannot_start_a_run(fault):
    data = profile()
    if fault == "missing_watch":
        data["cases"][0]["source"] = "missing"
    elif fault == "out_of_bounds":
        data["cases"][0]["destination"] = [640, 180]
    elif fault == "satisfied_condition":
        data["cases"][0]["before"]["condition"] = True
    else:
        data["cases"][0]["name"] = "../outside"
    with pytest.raises(ValueError):
        Profile.model_validate(data)
