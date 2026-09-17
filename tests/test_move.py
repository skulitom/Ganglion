"""Track watches, the track reflex with the connectome, and continuous move programs."""
import numpy as np
import pytest

from ganglion.brain.shadow import ShadowWorker
from ganglion.core.output import MemoryOutput
from ganglion.core.runtime import Runtime, RuntimeErrorWithCode
from ganglion.core.schema import ArmSpec, IntentSpec, WatchSpec
from ganglion.percepts.track import detect_track, init_track
from test_runtime import Clock
from test_supervised import Proposer

LAYOUT = {"hwnd": 0, "pid": 0, "class": "simulation", "rect": [0, 0, 640, 360]}


def textured(frame, x, y, size=28):
    """A checkered square centred at (x, y): enough texture for a template."""
    x0, y0 = int(round(x)) - size // 2, int(round(y)) - size // 2
    for i in range(size):
        for j in range(size):
            if 0 <= x0 + j < 640 and 0 <= y0 + i < 360:
                frame[y0 + i, x0 + j] = (200, 200, 200) if (i // 4 + j // 4) % 2 == 0 else (40, 60, 160)


def frame_at(x, y):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (25, 20, 18)
    if x is not None:
        textured(frame, x, y)
    return frame


def test_track_follows_a_moving_crop_and_reports_loss():
    spec = WatchSpec(name="t", snapshot_id="s", region=[0, 0, 640, 360], kind="track",
                     template_region=[286, 166, 28, 28], search_px=60)
    state = init_track(frame_at(300, 180), (286, 166, 28, 28), spec)
    assert (state["x"], state["y"]) == (300, 180)
    for step in range(1, 12):
        found = detect_track(frame_at(300 + 5 * step, 180 + 2 * step), spec, state)
        assert found and abs(found["x"] - (300 + 5 * step)) <= 2 and abs(found["y"] - (180 + 2 * step)) <= 2
        assert found["score"] > 0.8
    for _ in range(12):
        gone = detect_track(frame_at(None, None), spec, state)
        assert gone is None
    assert "template" not in state          # lost for good after repeated misses
    with pytest.raises(ValueError):
        init_track(frame_at(None, None), (10, 10, 28, 28), spec)   # no texture


class World:
    """A first-person plant: a textured target whose screen position moves opposite to look
    deltas, plus its own drift while the view is still."""
    def __init__(self, x, y):
        self.x, self.y = x, y

    def look(self, dx, dy):
        self.x -= dx
        self.y -= dy


def make_runtime(world, predictor=None):
    clock = Clock()
    output = MemoryOutput(clock, on_look=world.look)
    runtime = Runtime(clock, output)
    if predictor is not None:
        runtime.shadow = ShadowWorker(predictor, runtime.ledger, clock=clock)
    runtime.observe(frame_at(world.x, world.y), clock(), 1, LAYOUT)
    runtime.claim("agent", 20)
    return clock, output, runtime


def observe(runtime, clock, world, seq):
    runtime.observe(frame_at(world.x, world.y), clock(), seq, LAYOUT)
    runtime.tick()


def test_track_reflex_follows_the_thing_that_moved_with_the_connectome():
    world = World(430, 230)
    predictor = Proposer(sign=1)
    clock, output, runtime = make_runtime(world, predictor)
    try:
        snap = runtime.snapshot()["id"]
        wid = runtime.add_watch("agent", WatchSpec(name="motion", snapshot_id=snap, region=LAYOUT["rect"],
                                                    kind="motion", min_pixels=100, lag_ms=30))["watch_id"]
        runtime.arm("agent", ArmSpec(watch_id=wid, response="track", trigger="present", cooldown_ms=300,
                                     align={"controller": "connectome", "tolerance_px": 8, "settle_ms": 20,
                                            "gain": 1.0, "max_step": 40, "fire": {"hold_ms": 40, "repeat": 1}}))
        seq = 2
        for _ in range(400):
            clock.advance(.01)
            if runtime.intent is None:
                world.x += 2            # the target drifts while we are still: motion
            observe(runtime, clock, world, seq)
            seq += 1
            if runtime.intent is not None:
                # Let the shadow worker finish the inference for this tick before the next one.
                import time as _t
                until = _t.perf_counter() + 0.5
                while runtime.shadow.predictions + runtime.shadow.discarded < predictor.calls and _t.perf_counter() < until:
                    _t.sleep(.001)
                _t.sleep(.001)
            if runtime.intent is not None and not runtime.intent.active:
                break
        intent = runtime.intent
        assert intent is not None and intent.spec.program == "align" and intent.spec.controller == "connectome"
        assert intent.phase == "completed" and intent.reason == "aligned_and_fired", intent.status()
        assert abs(world.x - 320) <= 24 and abs(world.y - 180) <= 24   # the tracked crop centre, not the square centre
        assert intent.neural_commands > 0
        assert intent.cleanup_watch not in runtime.watches      # the spawned tracker is gone
        events = runtime.ledger.read(limit=400)["events"]
        assert any(e["kind"] == "watch_added" and e.get("watch_kind") == "track" for e in events)
        assert any(e["kind"] == "look_done" and e.get("controller") == "connectome" for e in events)
    finally:
        if runtime.shadow:
            runtime.shadow.close()


def test_move_renews_keys_until_its_condition_and_can_run_beside_align():
    world = World(320, 180)
    clock, output, runtime = make_runtime(world)
    snap = runtime.snapshot()["id"]
    goal = runtime.add_watch("agent", WatchSpec(name="goal", snapshot_id=snap, region=LAYOUT["rect"],
                                                 color_rgb=[200, 200, 200], tolerance=20, min_pixels=50))["watch_id"]
    move = runtime.start_intent("agent", IntentSpec(program="move", keys=["w", "lshift"], timeout_seconds=2,
                                                    until={"watch_id": goal, "present": False}))
    assert move["program"] == "move" and runtime.locomotion.active
    # A pointer program may run at the same time: keys and pointer are different actuators.
    runtime.start_intent("agent", IntentSpec(program="align", watch_id=goal, tolerance_px=30, settle_ms=20))
    seq = 2
    for _ in range(50):
        clock.advance(.01)
        observe(runtime, clock, world, seq)
        seq += 1
    holds = [c for c in output.commands if c.get("keys") == ["w", "lshift"] and "hold_until" in c]
    assert len(holds) >= 4 and all(c["hold_until"] - c["deadline"] <= 0.35 for c in holds)
    assert runtime.intent.phase == "completed"            # align finished while the move went on
    assert runtime.locomotion.active
    world.x = None                                         # the goal disappears: the move stops
    for _ in range(10):
        clock.advance(.01)
        observe(runtime, clock, world, seq)
        seq += 1
    assert runtime.locomotion.phase == "completed" and runtime.locomotion.reason == "condition_observed"
    assert output.commands[-1].get("keys") == ["w", "lshift"] and "hold_until" not in output.commands[-1]   # explicit release


def test_move_duration_cancel_and_validation():
    world = World(None, None)
    clock, output, runtime = make_runtime(world)
    with pytest.raises(Exception):
        IntentSpec(program="move")
    with pytest.raises(Exception):
        IntentSpec(program="move", keys=["w"], watch_id="x")
    runtime.start_intent("agent", IntentSpec(program="move", keys=["a"], timeout_seconds=0.3))
    with pytest.raises(RuntimeErrorWithCode) as excinfo:
        runtime.start_intent("agent", IntentSpec(program="move", keys=["d"], timeout_seconds=0.3))
    assert excinfo.value.code == "keys_busy"
    for seq in range(2, 40):
        clock.advance(.01)
        observe(runtime, clock, world, seq)
    assert runtime.locomotion.phase == "completed" and runtime.locomotion.reason == "duration"
    second = runtime.start_intent("agent", IntentSpec(program="move", keys=["d"], timeout_seconds=5))
    runtime.cancel("agent", second["intent_id"])
    assert runtime.locomotion.phase == "cancelled"
    assert runtime.status()["locomotion"]["phase"] == "cancelled"
    runtime.start_intent("agent", IntentSpec(program="move", keys=["s"], timeout_seconds=5))
    runtime.halt()
    assert runtime.locomotion.phase == "cancelled" and runtime.state == "halted"
