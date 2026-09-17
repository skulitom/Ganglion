from dataclasses import asdict, replace
import threading
import time

import pytest

from ganglion.brain.haltere_cursor import channels
from ganglion.brain.shadow import MotorSample, ShadowWorker
from ganglion.core.ledger import Ledger


def sample():
    now = time.perf_counter()
    return MotorSample("intent", "reach", 1, 1, now, now, now+1,
                       (40, 50), (300, 200), (50, 56), (0, 0, 640, 360), 1200, .01)


def eventually(predicate):
    until = time.perf_counter() + 2
    while not predicate():
        if time.perf_counter() >= until:
            raise AssertionError("Worker did not report its outcome")
        time.sleep(.002)


class Predictor:
    metadata = {"type": "test"}
    def __init__(self):
        self.calls = []
        self.resets = 0
    def reset(self):
        self.resets += 1
    def predict(self, value):
        self.calls.append(value)
        return {"point": [-100, -100], "raw_actions": [-1, -1, 0, 0]}


class Blocked(Predictor):
    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()
    def predict(self, value):
        self.entered.set()
        self.release.wait(3)
        return super().predict(value)


def test_actual_motor_commands_ignore_even_a_stalled_shadow_model():
    from test_reach import setup, observe
    def run(predictor=None):
        rig = setup(click=True)
        clock, output, runtime, _, _ = rig
        if predictor:
            runtime.shadow = ShadowWorker(predictor, runtime.ledger, clock=clock)
        try:
            for seq in range(2, 152):
                clock.advance(.01)
                observe(rig, seq)
                if seq == 4 and predictor:
                    assert predictor.entered.wait(1)
                if not runtime.intent.active:
                    break
            assert runtime.intent.phase == "completed"
            return [(cmd.get("point"), cmd.get("x"), cmd.get("y"), cmd.get("hold_ms")) for cmd in output.commands]
        finally:
            if predictor:
                predictor.release.set()
                eventually(lambda: runtime.shadow.discarded > 0)
                assert runtime.shadow.predictions == 0
                runtime.shadow.close()
    assert run(Blocked()) == run()


def test_latest_mailbox_replaces_backlog_and_invalidates_inflight_prediction():
    predictor, ledger = Blocked(), Ledger()
    worker = ShadowWorker(predictor, ledger)
    try:
        first = sample()
        worker.submit(first)
        assert predictor.entered.wait(1)
        worker.submit(replace(sample(), observation_id=2))
        worker.submit(replace(sample(), observation_id=3))
        assert worker.status()["replaced"] == 1
        worker.invalidate()
        predictor.release.set()
        eventually(lambda: worker.discarded == 1)
        assert len(predictor.calls) == 1
        assert all(e["kind"] != "shadow_prediction" for e in ledger.read(limit=200)["events"])
    finally:
        predictor.release.set()
        worker.close()


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_shadow_failure_is_isolated_and_disables_future_inference(failure):
    class Broken(Predictor):
        def predict(self, value):
            self.calls.append(value)
            if failure == "exception":
                raise RuntimeError("inference failed")
            return {"point": [float("nan"), 0]}
    predictor, ledger = Broken(), Ledger()
    worker = ShadowWorker(predictor, ledger)
    try:
        worker.submit(sample())
        eventually(lambda: worker.failed is not None)
        worker.submit(sample())
        assert len(predictor.calls) == 1
        assert worker.status()["state"] == "failed"
        event = ledger.read()["events"][0]
        assert event["kind"] == "shadow_failed" and not event["actuation_authority"]
    finally:
        worker.close()


def test_shadow_resets_on_new_binding_and_drops_stale_sample_before_inference():
    predictor, ledger = Predictor(), Ledger()
    worker = ShadowWorker(predictor, ledger)
    try:
        worker.submit(replace(sample(), sampled=time.perf_counter()-1))
        eventually(lambda: worker.discarded == 1)
        assert not predictor.calls
        for i, stage in enumerate(("approaching", "dragging"), start=1):
            worker.submit(replace(sample(), stage=stage))
            eventually(lambda: worker.predictions == i)
        assert predictor.resets == 2
        assert all(not e["actuation_authority"] for e in ledger.read()["events"])
    finally:
        worker.close()


def test_channel_adapter_is_translation_invariant_and_has_no_game_semantics():
    original = sample()
    translated = replace(original, cursor=(140, 250), goal=(400, 400), rect=(100, 200, 640, 360))
    assert channels(original) == channels(translated)
    moving = replace(original, cursor=(52, 44), submitted=original.submitted+.01)
    result = channels(moving, original)
    assert result["haltere"][0] > 0 and result["haltere"][1] < 0
    assert result["jo"] == result["haltere"]
    assert result["lptc"] == [0]*6  # No measured optic flow exists yet.
    assert all(-1 <= v <= 1 for values in result.values() for v in values)


def test_frozen_replay_preserves_neural_reset_boundaries():
    from ganglion.brain.replay import compare
    class Stateful(Predictor):
        def reset(self):
            super().reset()
            self.value = 0
        def predict(self, value):
            self.value += 1
            return {"point": [self.value, self.value], "raw_actions": [self.value]*4}
    events = [{"kind": "shadow_prediction", "sensor_input": asdict(sample()), "reset": reset,
               "point": [n, n], "raw_actions": [n]*4}
              for reset, n in ((True, 1), (False, 2), (True, 1))]
    predictor = Stateful()
    result = compare(predictor, events)
    assert result["same_recorded_point_fraction"] == 1
    assert result["max_recorded_raw_action_error"] == 0
    assert predictor.resets == 2 and not result["promoted"]
    with pytest.raises(ValueError, match="sensor inputs"):
        compare(predictor, [{"kind": "shadow_prediction"}])
