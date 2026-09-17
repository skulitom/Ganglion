"""Supervised connectome authority: proposals drive the cursor only inside the envelope."""
from math import hypot
import time

import numpy as np
import pytest

from ganglion.brain.shadow import ShadowWorker
from ganglion.core.output import MemoryOutput
from ganglion.core.reach import supervise
from ganglion.core.runtime import Runtime, RuntimeErrorWithCode
from ganglion.core.schema import IntentSpec, WatchSpec
from test_runtime import Clock
from test_reach import observe


class Proposer:
    """Fake model: a unit velocity toward (sign=1) or away from (sign=-1) the goal."""
    metadata = {"type": "test", "desktop_trained": False}

    def __init__(self, sign=1):
        self.sign, self.calls, self.resets = sign, 0, 0

    def reset(self):
        self.resets += 1

    def predict(self, sample):
        self.calls += 1
        dx, dy = (sample.goal[i] - sample.cursor[i] for i in range(2))
        norm = hypot(dx, dy) or 1.0
        v = [self.sign * dx / norm, self.sign * dy / norm]
        dt = min(.02, sample.dt)
        return {"point": [sample.cursor[0] + v[0] * sample.speed * dt, sample.cursor[1] + v[1] * sample.speed * dt],
                "raw_actions": v + [0.0, 0.0]}


def rig(predictor, controller="connectome", **params):
    clock = Clock()
    output = MemoryOutput(clock)
    runtime = Runtime(clock, output)
    if predictor is not None:
        runtime.shadow = ShadowWorker(predictor, runtime.ledger, clock=clock)
    layout = {"hwnd": 0, "pid": 0, "class": "simulation", "rect": [0, 0, 640, 360]}
    runtime.observe(np.zeros((360, 640, 3), dtype=np.uint8), clock(), 1, layout)
    runtime.claim("agent", 10)
    wid = runtime.add_watch("agent", WatchSpec(name="goal", snapshot_id=runtime.snapshot()["id"],
        region=layout["rect"], color_rgb=[40, 220, 120]))["watch_id"]
    runtime.start_intent("agent", IntentSpec(watch_id=wid, controller=controller, **params))
    return clock, output, runtime, layout, wid


def drive(r, predictor):
    clock, output, runtime, _, _ = r
    try:
        for seq in range(2, 200):
            clock.advance(.01)
            observe(r, seq)
            if predictor is not None:
                # Let the worker finish the inference for this tick before the next one.
                until = time.perf_counter() + 1
                while runtime.shadow.predictions + runtime.shadow.discarded < predictor.calls - 0 and time.perf_counter() < until:
                    time.sleep(.001)
                time.sleep(.002)
            if not runtime.intent.active:
                break
    finally:
        if runtime.shadow:
            runtime.shadow.close()
    return runtime.intent


def test_supervise_accepts_progress_and_rejects_retreat_or_nonsense():
    rect = (0, 0, 640, 360)
    assert supervise((100, 100), (300, 100), (1.0, 0.0), .01, 1200, rect) == (112, 100)
    assert supervise((100, 100), (300, 100), (-1.0, 0.0), .01, 1200, rect) is None
    assert supervise((100, 100), (300, 100), (0.0, 0.0), .01, 1200, rect) is None
    assert supervise((100, 100), (300, 100), (float("nan"), 0.0), .01, 1200, rect) is None
    assert supervise((100, 100), (300, 100), [1.0], .01, 1200, rect) is None
    # A large proposal is clamped to the speed limit, never amplified.
    assert supervise((100, 100), (300, 100), (50.0, 0.0), .01, 1200, rect) == (112, 100)
    # Inside tolerance a tiny drift that does not leave the goal is allowed.
    assert supervise((300, 100), (300, 100), (0.0, 0.0), .01, 1200, rect) == (300, 100)
    assert supervise((300, 100), (300, 100), (1.0, 0.0), .01, 1200, rect) is None


def test_connectome_controller_requires_a_loaded_model():
    with pytest.raises(RuntimeErrorWithCode) as excinfo:
        rig(None, click=True)
    assert excinfo.value.code == "controller_unavailable"


def test_helpful_proposals_drive_the_cursor_and_the_reach_completes():
    predictor = Proposer(sign=1)
    r = rig(predictor, click=True)
    intent = drive(r, predictor)
    assert intent.phase == "completed" and intent.error_px <= 6
    status = intent.status()
    assert status["actuation_authority"] == "supervised_connectome" and status["controller"] == "connectome"
    assert status["neural_commands"] > 5
    events = r[2].ledger.read(limit=200)["events"]
    controllers = {e.get("controller") for e in events if e["kind"] == "pointer_feedback"}
    assert "connectome" in controllers


def test_harmful_proposals_are_overridden_and_the_reach_still_completes():
    predictor = Proposer(sign=-1)
    r = rig(predictor, click=True)
    intent = drive(r, predictor)
    assert intent.phase == "completed" and intent.error_px <= 6
    status = intent.status()
    assert status["neural_commands"] == 0 and status["overridden_commands"] > 5
    events = r[2].ledger.read(limit=200)["events"]
    assert all(e.get("controller") != "connectome" for e in events if e["kind"] == "pointer_feedback")


def test_deterministic_intents_ignore_proposals_entirely():
    predictor = Proposer(sign=1)
    r = rig(predictor, controller="deterministic", click=True)
    intent = drive(r, predictor)
    assert intent.phase == "completed"
    status = intent.status()
    assert status["neural_commands"] == 0 and status["overridden_commands"] == 0
    assert status["actuation_authority"] == "deterministic"
