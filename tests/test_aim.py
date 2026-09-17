"""Keys, relative look, button holds, motion watches and the align program."""
import numpy as np
import pytest

from ganglion.core.output import ClickMachine, MemoryOutput
from ganglion.core.runtime import Runtime, RuntimeErrorWithCode
from ganglion.core.schema import ArmSpec, InputSpec, IntentSpec, WatchSpec
from ganglion.percepts.motion import detect_motion
from test_runtime import Clock


class Plant:
    """Fake actuator: records keys, buttons and relative moves."""
    def __init__(self):
        self.keys, self.buttons, self.moves, self.log = set(), set(), [], []

    def move_abs(self, x, y):
        self.log.append(("abs", x, y))

    def move_rel(self, dx, dy):
        self.moves.append((dx, dy))
        self.log.append(("rel", dx, dy))

    def cursor_pos(self):
        return (0, 0)

    def key(self, name, down):
        (self.keys.add if down else self.keys.discard)(name)
        self.log.append(("key", name, down))

    def button(self, name, down):
        (self.buttons.add if down else self.buttons.discard)(name)
        self.log.append(("button", name, down))

    def release_all(self):
        for k in list(self.keys):
            self.key(k, False)
        for b in list(self.buttons):
            self.button(b, False)


def machine():
    clock, plant, events = Clock(), Plant(), []
    return clock, plant, events, ClickMachine(plant, clock, lambda *a: None, events.append)


def test_keys_hold_extend_and_release_on_their_own_deadline():
    clock, plant, events, m = machine()
    m.key({"command_id": "walk", "target": {}, "deadline": clock() + .05, "keys": ["w", "a"], "hold_until": clock() + .3})
    assert plant.keys == {"w", "a"} and events[-1]["kind"] == "keys_held"
    clock.advance(.2)
    m.key({"command_id": "walk2", "target": {}, "deadline": clock() + .05, "keys": ["w"], "hold_until": clock() + .3})
    assert events[-1]["extended"] == ["w"]
    clock.advance(.15)
    m.tick()
    assert plant.keys == {"w"}                       # a expired, w was extended
    assert events[-1]["kind"] == "keys_released" and events[-1]["keys"] == ["a"]
    clock.advance(.2)
    m.tick()
    assert not plant.keys and events[-1]["keys"] == ["w"]
    # A hold can never exceed one second per command, whatever the producer asks.
    m.key({"command_id": "long", "target": {}, "deadline": clock() + .05, "keys": ["s"], "hold_until": clock() + 30})
    assert m.keys["s"][1] <= clock() + 1.0


def test_look_spreads_deltas_over_time_and_button_holds_release():
    clock, plant, events, m = machine()
    m.look({"command_id": "turn", "target": {}, "deadline": clock() + .05, "dx": 25, "dy": -5, "chunks": 5})
    assert plant.moves == [(5, -1)] and events[-1]["kind"] == "look_submitted"
    clock.advance(.011)
    m.tick()
    assert len(plant.moves) == 2
    clock.advance(.05)
    m.tick()
    assert sum(dx for dx, _ in plant.moves) == 25 and sum(dy for _, dy in plant.moves) == -5
    assert events[-1]["kind"] == "look_done"
    m.button({"command_id": "fire", "target": {}, "deadline": clock() + .05, "button": "left", "hold_until": clock() + .1})
    assert plant.buttons == {"left"} and events[-1]["kind"] == "input_submitted"
    m.button({"command_id": "fire2", "target": {}, "deadline": clock() + .05, "button": "left", "hold_until": clock() + .1})
    assert events[-1]["reason"] == "pointer_busy"
    clock.advance(.11)
    m.tick()
    assert not plant.buttons and events[-1]["kind"] == "input_released" and events[-1]["button"] == "left"


def test_halt_releases_keys_buttons_and_queued_looks():
    clock, plant, events, m = machine()
    m.key({"command_id": "walk", "target": {}, "deadline": clock() + .05, "keys": ["w"], "hold_until": clock() + .5})
    m.button({"command_id": "fire", "target": {}, "deadline": clock() + .05, "button": "right", "hold_until": clock() + .5})
    m.look({"command_id": "turn", "target": {}, "deadline": clock() + .05, "dx": 40, "dy": 0, "chunks": 4})
    m.halt()
    assert not plant.keys and not plant.buttons and not m.looks
    kinds = [(e["kind"], e["command_id"]) for e in events[-3:]]
    assert ("keys_released", "walk") in kinds and ("input_released", "fire") in kinds and ("input_cancelled", "turn") in kinds


def test_motion_detector_reports_change_only_between_still_frames():
    spec = WatchSpec(name="m", snapshot_id="s", region=[0, 0, 200, 100], kind="motion", min_pixels=50, lag_ms=60, persist=1)
    state = {}
    frames = []
    for t in range(6):
        f = np.zeros((100, 200, 3), dtype=np.uint8)
        f[30:60, 20 + 8 * t:50 + 8 * t] = 200
        frames.append(f)
    assert detect_motion(frames[0], spec, state, captured=0.0, moving=False) is None      # no reference yet
    assert detect_motion(frames[1], spec, state, captured=0.03, moving=False) is None      # reference too young
    found = detect_motion(frames[2], spec, state, captured=0.07, moving=False)
    assert found and found["pixels"] >= 50 and 20 < found["x"] < 80
    assert detect_motion(frames[3], spec, state, captured=0.10, moving=True) is None       # own motion: suppressed
    assert not state["history"]
    assert detect_motion(frames[3], spec, state, captured=0.13, moving=False) is None      # history restarted
    assert detect_motion(frames[3], spec, state, captured=0.20, moving=False) is None      # still: nothing moved


class Camera:
    """Simulated first-person view: a target whose screen position shifts opposite to look deltas."""
    def __init__(self, x, y, counts_per_px=1.0):
        self.x, self.y, self.k = x, y, counts_per_px

    def look(self, dx, dy):
        self.x -= dx / self.k
        self.y -= dy / self.k


def rig(camera, *, kind="color", fire=None, **params):
    clock = Clock()
    output = MemoryOutput(clock, on_look=camera.look)
    runtime = Runtime(clock, output)
    layout = {"hwnd": 0, "pid": 0, "class": "simulation", "rect": [0, 0, 640, 360]}
    runtime.observe(np.zeros((360, 640, 3), dtype=np.uint8), clock(), 1, layout)
    runtime.claim("agent", 10)
    spec = dict(name="enemy", snapshot_id=runtime.snapshot()["id"], region=layout["rect"])
    if kind == "color":
        spec["color_rgb"] = [40, 220, 120]
    else:
        spec.update(kind="motion", min_pixels=100, lag_ms=30)
    wid = runtime.add_watch("agent", WatchSpec(**spec))["watch_id"]
    return clock, output, runtime, layout, wid


def frame_with(camera, layout):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    x, y = int(round(camera.x)), int(round(camera.y))
    if 0 <= x < 640 and 0 <= y < 360:
        frame[max(0, y - 12):y + 12, max(0, x - 12):x + 12] = [120, 220, 40]
    return frame


def run_align(camera, seconds=2.0, **params):
    clock, output, runtime, layout, wid = rig(camera, **{k: v for k, v in params.items() if k == "kind"})
    spec = {k: v for k, v in params.items() if k != "kind"}
    runtime.start_intent("agent", IntentSpec(program="align", watch_id=wid, **spec))
    seq = 2
    while clock() - 10 < seconds and runtime.intent.active:
        clock.advance(.01)
        runtime.observe(frame_with(camera, layout), clock(), seq, layout)
        runtime.tick()
        seq += 1
    return runtime, output


def test_align_turns_the_view_until_the_target_is_centred():
    camera = Camera(500, 100, counts_per_px=2.0)
    runtime, output = run_align(camera, gain=2.0, tolerance_px=4, settle_ms=30)
    intent = runtime.intent
    assert intent.phase == "completed" and intent.reason == "aligned"
    assert abs(camera.x - 320) <= 4 and abs(camera.y - 180) <= 4
    looks = [c for c in output.commands if "dx" in c]
    assert looks and all(abs(c["dx"]) <= 60 and abs(c["dy"]) <= 60 for c in looks)
    assert intent.status()["applied_delta"][0] > 0 and intent.commands == len(looks)


def test_align_fires_repeatedly_while_the_target_stays_and_reports_loss_after_fire():
    camera = Camera(340, 190)
    runtime, output = run_align(camera, gain=1.0, tolerance_px=30, settle_ms=20,
                                fire={"button": "left", "hold_ms": 50, "repeat": 2, "interval_ms": 100})
    intent = runtime.intent
    assert intent.phase == "completed" and intent.reason == "aligned_and_fired"
    fires = [c for c in output.commands if c.get("button") == "left"]
    assert len(fires) == 2 and intent.shots == 2
    # A target that vanishes after a shot ends the program as completed, not failed.
    camera = Camera(330, 185)
    clock, output, runtime, layout, wid = rig(camera)
    runtime.start_intent("agent", IntentSpec(program="align", watch_id=wid, tolerance_px=30, settle_ms=20,
                                             absence_ms=100, fire={"hold_ms": 50, "repeat": 5, "interval_ms": 50}))
    seq = 2
    for _ in range(80):
        clock.advance(.01)
        if seq == 12:
            camera.x = -100      # gone
        runtime.observe(frame_with(camera, layout), clock(), seq, layout)
        runtime.tick()
        seq += 1
        if not runtime.intent.active:
            break
    assert runtime.intent.phase == "completed" and runtime.intent.reason == "target_lost_after_fire"
    assert runtime.intent.shots >= 1


def test_align_fails_on_a_target_never_seen_and_refuses_points_outside():
    camera = Camera(-50, -50)
    runtime, output = run_align(camera, absence_ms=100)
    assert runtime.intent.phase == "failed" and runtime.intent.reason == "target_lost"
    clock, output, runtime, layout, wid = rig(Camera(300, 100))
    with pytest.raises(RuntimeErrorWithCode) as excinfo:
        runtime.start_intent("agent", IntentSpec(program="align", watch_id=wid, point=[700, 100]))
    assert excinfo.value.code == "point_outside_target"


def test_inputs_hold_keys_look_and_fire_with_self_motion_noted():
    clock, output, runtime, layout, wid = rig(Camera(300, 100))
    snap = runtime.snapshot()["id"]
    runtime.input("agent", InputSpec(action="hold", snapshot_id=snap, keys=["w", "a"], hold_ms=400))
    assert runtime.motion_until >= clock() + .4
    assert output.commands[-1]["keys"] == ["w", "a"] and not runtime.pending    # keys never block the pointer
    runtime.input("agent", InputSpec(action="look", snapshot_id=snap, delta=[120, -20], spread_ms=100))
    assert output.commands[-1]["chunks"] == 10
    runtime.tick()
    runtime.input("agent", InputSpec(action="button", snapshot_id=snap, button="right", hold_ms=80))
    assert output.commands[-1]["button"] == "right"
    runtime.tick()
    runtime.input("agent", InputSpec(action="key", snapshot_id=snap, key="e", hold_ms=30))
    assert output.commands[-1]["keys"] == ["e"]
    kinds = [e["kind"] for e in runtime.ledger.read(limit=200)["events"]]
    assert "keys_held" in kinds and "look_done" in kinds and "input_released" in kinds
    with pytest.raises(Exception):
        InputSpec(action="look", snapshot_id=snap)


def test_motion_watch_is_blind_while_own_commands_move_and_an_align_reflex_engages():
    camera = Camera(400, 200)
    clock, output, runtime, layout, wid = rig(camera, kind="motion")
    rid = runtime.arm("agent", ArmSpec(watch_id=wid, response="align", trigger="appear",
                                       align={"tolerance_px": 10, "settle_ms": 20, "gain": 1.0}))["reflex_id"]
    snap = runtime.snapshot()["id"]
    seq = 2
    # Own movement: the target drifts but the watch must stay blind.
    runtime.input("agent", InputSpec(action="hold", snapshot_id=snap, keys=["w"], hold_ms=200))
    for _ in range(20):
        clock.advance(.01)
        camera.x += 3
        runtime.observe(frame_with(camera, layout), clock(), seq, layout)
        runtime.tick()
        seq += 1
    assert not runtime.watches[wid].present and runtime.intent is None
    # Still: the drifting target is motion, the reflex engages, and align centres it.
    clock.advance(.4)
    for _ in range(120):
        clock.advance(.01)
        if runtime.intent is None:
            camera.x += 3
        runtime.observe(frame_with(camera, layout), clock(), seq, layout)
        runtime.tick()
        seq += 1
        if runtime.intent is not None and not runtime.intent.active:
            break
    assert runtime.intent is not None and runtime.intent.spec.program == "align"
    events = runtime.ledger.read(limit=300)["events"]
    assert any(e["kind"] == "reflex_fired" and e.get("response") == "align" for e in events)
    assert runtime.reflexes[rid].fires == 1
