"""Proposal freshness is timed from the observation; neural time follows elapsed time."""
import time

from ganglion.brain.haltere_cursor import neural_steps
from ganglion.brain.shadow import MotorSample, ShadowWorker
from ganglion.core.ledger import Ledger


class Proposer:
    metadata = {"type": "test"}

    def __init__(self, steps=1):
        self.steps, self.calls = steps, 0

    def reset(self):
        pass

    def predict(self, sample):
        self.calls += 1
        return {"point": [1, 1], "raw_actions": [0.5, -0.25, 0, 0], "neural_steps": self.steps}


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def eventually(predicate):
    until = time.perf_counter() + 2
    while not predicate():
        if time.perf_counter() >= until:
            raise AssertionError("worker did not finish")
        time.sleep(.002)


def test_proposal_reuse_is_bounded_by_observation_age_not_inference_completion():
    clock, ledger, predictor = Clock(), Ledger(), Proposer()
    worker = ShadowWorker(predictor, ledger, clock=clock)
    try:
        sampled = clock.now - .04                       # the observation is already 40 ms old
        sample = MotorSample("i", "reach", 1, 7, sampled, clock.now, clock.now + 5,
                             (0, 0), (100, 0), (10, 0), (0, 0, 640, 360), 1200, .01)
        worker.submit(sample)
        eventually(lambda: worker.predictions == 1)
        binding = ("i", "reach", 1)
        assert worker.proposal(binding, clock.now) == [0.5, -0.25]
        clock.now += .009                               # 49 ms after the observation: still usable
        assert worker.proposal(binding, clock.now) is not None
        clock.now += .002                               # 51 ms after the observation: expired,
        assert worker.proposal(binding, clock.now) is None   # although inference finished 11 ms ago
        assert worker.proposal(("other", "reach", 1), clock.now - .01) is None
        event = [e for e in ledger.read(limit=50)["events"] if e["kind"] == "shadow_prediction"][0]
        assert event["neural_steps"] == 1
    finally:
        worker.close()


def test_neural_steps_track_elapsed_wall_time_and_stay_bounded():
    assert neural_steps(None) == 1
    assert neural_steps(0.0) == 1
    assert neural_steps(0.004) == 1
    assert neural_steps(0.01) == 1
    assert neural_steps(0.015) == 2
    assert neural_steps(0.03) == 2                      # capped: a late loop cannot afford more
    assert neural_steps(0.049) == 2
    assert neural_steps(0.2) == 2                       # beyond the reset window a reset applies
    assert neural_steps(0.03, dt=0.005, max_steps=20) == 6


def test_predictor_reported_steps_reach_the_ledger():
    clock, ledger, predictor = Clock(), Ledger(), Proposer(steps=3)
    worker = ShadowWorker(predictor, ledger, clock=clock)
    try:
        sample = MotorSample("i", "reach", 1, 1, clock.now, clock.now, clock.now + 5,
                             (0, 0), (100, 0), (10, 0), (0, 0, 640, 360), 1200, .01)
        worker.submit(sample)
        eventually(lambda: worker.predictions == 1)
        event = [e for e in ledger.read(limit=50)["events"] if e["kind"] == "shadow_prediction"][0]
        assert event["neural_steps"] == 3
    finally:
        worker.close()
