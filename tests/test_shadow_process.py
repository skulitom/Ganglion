"""The out-of-process shadow worker keeps the in-process contract: mailbox, freshness, ledger, isolation."""
import time

import pytest

from ganglion.brain.shadow import MotorSample
from ganglion.brain.shadow_process import ShadowProcess
from ganglion.core.ledger import Ledger


class Proposer:
    metadata = {"type": "test", "desktop_trained": False}

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def reset(self):
        pass

    def predict(self, sample):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("model broke")
        return {"point": [1, 1], "raw_actions": [0.5, -0.25, 0, 0], "neural_steps": 2}


def make_proposer():
    return Proposer()


def make_failing_proposer():
    return Proposer(fail_after=1)


def broken_factory():
    raise ValueError("no checkpoint")


def sample(now, **overrides):
    fields = dict(intent_id="i", stage="reach", layout_revision=1, observation_id=1, sampled=now, submitted=now,
                  expires=now + 5, cursor=(0, 0), goal=(100, 0), reference=(10, 0), rect=(0, 0, 640, 360),
                  speed=1200, dt=.01)
    fields.update(overrides)
    return MotorSample(**fields)


def eventually(predicate, seconds=10):
    until = time.perf_counter() + seconds
    while not predicate():
        if time.perf_counter() >= until:
            raise AssertionError("the shadow process did not answer in time")
        time.sleep(.01)


def test_process_predicts_records_and_isolates_like_the_thread_worker():
    ledger = Ledger()
    shadow = ShadowProcess(make_proposer, ledger)
    try:
        assert shadow.status()["state"] == "ready" and shadow.status()["process"] is True
        assert shadow.metadata["type"] == "test"
        now = time.perf_counter()
        shadow.submit(sample(now))
        eventually(lambda: shadow.predictions == 1)
        assert shadow.proposal(("i", "reach", 1), time.perf_counter()) == [0.5, -0.25]
        assert shadow.proposal(("other", "reach", 1), time.perf_counter()) is None
        events = [e for e in ledger.read(limit=50)["events"] if e["kind"] == "shadow_prediction"]
        assert events and events[0]["neural_steps"] == 2 and events[0]["actuation_authority"] is False
        assert tuple(events[0]["sensor_input"]["goal"]) == (100, 0)
        shadow.invalidate()
        assert shadow.proposal(("i", "reach", 1), time.perf_counter()) is None
        shadow.submit(sample(time.perf_counter() - 1))          # already stale: discarded in the child
        eventually(lambda: shadow.discarded >= 1)
        assert shadow.predictions == 1
    finally:
        shadow.close()
    assert not shadow.process.is_alive() and shadow.status()["state"] == "closed"


def test_model_failure_in_the_child_is_reported_and_never_raises_in_the_runtime():
    ledger = Ledger()
    shadow = ShadowProcess(make_failing_proposer, ledger)
    try:
        shadow.submit(sample(time.perf_counter()))
        eventually(lambda: shadow.predictions == 1)
        shadow.submit(sample(time.perf_counter(), observation_id=2))
        eventually(lambda: shadow.failed is not None)
        assert "model broke" in shadow.failed and shadow.status()["state"] == "failed"
        assert shadow.proposal(("i", "reach", 1), time.perf_counter()) is None
        assert any(e["kind"] == "shadow_failed" for e in ledger.read(limit=50)["events"])
        shadow.submit(sample(time.perf_counter(), observation_id=3))   # ignored once failed
    finally:
        shadow.close()


def test_a_factory_that_cannot_load_fails_construction_cleanly():
    with pytest.raises(RuntimeError, match="no checkpoint"):
        ShadowProcess(broken_factory, Ledger())
