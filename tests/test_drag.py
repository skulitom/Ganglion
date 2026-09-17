import numpy as np
import pytest

from ganglion.core.output import SimulatedOutput
from ganglion.core.runtime import Runtime, RuntimeErrorWithCode
from ganglion.core.schema import IntentSpec, WatchSpec
from test_runtime import Clock


class Plant:
    def __init__(self):
        self.cursor = (10, 60)
        self.held = False
        self.history = []
    def move_abs(self, x, y):
        self.cursor = (x, y)
        self.history.append(("move", self.cursor, self.held))
    def cursor_pos(self):
        return self.cursor
    def button(self, _, down):
        self.held = down
        self.history.append(("down" if down else "up", self.cursor, self.held))
    def release_all(self):
        if self.held:
            self.button("left", False)


def setup_drag(**params):
    clock, plant = Clock(), Plant()
    output = SimulatedOutput(clock, plant)
    r = Runtime(clock, output)
    layout = {"hwnd": 0, "pid": 0, "rect": [0, 0, 400, 200]}
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    r.observe(frame, clock(), 1, layout)
    r.claim("agent", 10)
    ids = []
    for name, rgb in (("source", [40, 220, 120]), ("condition", [60, 130, 240])):
        ids.append(r.add_watch("agent", WatchSpec(name=name, snapshot_id=r.snapshot()["id"],
                        region=layout["rect"], color_rgb=rgb))["watch_id"])
    spec = IntentSpec(program="drag", watch_id=ids[0], destination=[320, 60],
                      until={"watch_id": ids[1]}, **params)
    r.start_intent("agent", spec)
    return clock, plant, output, r, layout, ids


def sample(rig, seq, *, condition=False, source=True, started=None):
    c, _, _, r, layout, _ = rig
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    if source:
        frame[45:75, 85:115] = [120, 220, 40]
    if condition:
        frame[120:140, 100:200] = [240, 130, 60]
    r.observe(frame, c(), seq, layout, sample_started=started)
    r.tick()


def advance_to_hold(rig):
    for seq in range(2, 100):
        rig[0].advance(.01)
        sample(rig, seq)
        if rig[1].held:
            return seq + 1
    pytest.fail("drag did not press")


def test_drag_releases_on_condition_before_destination_and_verifies_after_release():
    rig = setup_drag()
    c, plant, _, r, _, _ = rig
    seq = advance_to_hold(rig)
    while plant.cursor[0] < 190:
        c.advance(.01)
        sample(rig, seq, source=False)
        seq += 1
    c.advance(.01)
    sample(rig, seq, source=False, condition=True)
    assert not plant.held and r.intent.stage == "verifying"
    assert r.intent.phase == "running" and plant.cursor[0] < 320
    for i in range(1, 10):
        c.advance(.01)
        sample(rig, seq + i, source=False, condition=True)
    assert r.intent.phase == "completed" and r.intent.condition_verified
    assert [e[0] for e in plant.history].count("down") == 1
    assert [e[0] for e in plant.history].count("up") == 1
    assert r.intent.status()["task_success_verified"] is False


def test_drag_arrival_alone_does_not_prove_visual_completion():
    rig = setup_drag(verification_seconds=.15)
    c, plant, _, r, _, _ = rig
    for seq in range(2, 190):
        c.advance(.01)
        sample(rig, seq)
        if not r.intent.active:
            break
    assert not plant.held and r.intent.reason == "condition_not_observed"


@pytest.mark.parametrize("cause", ["cancel", "lease", "timeout", "stale", "unwatch", "focus"])
def test_held_drag_releases_on_every_stop_path(cause):
    rig = setup_drag()
    c, plant, output, r, _, ids = rig
    advance_to_hold(rig)
    if cause == "cancel":
        r.cancel("agent", r.intent.id)
    elif cause == "lease":
        r.expires = c()
    elif cause == "timeout":
        r.intent.expires = c()
    elif cause == "stale":
        c.advance(.11)
    elif cause == "unwatch":
        r.remove("agent", ids[1], watch=True)
    else:
        def blocked(*args):
            raise ValueError("focus lost")
        output.machine.validate = blocked
        c.advance(.02)
    r.tick()
    r.tick()
    assert not plant.held and not r.intent.active and not r.pending


def test_preexisting_condition_is_rejected_without_a_button_press():
    rig = setup_drag()
    sample(rig, 2, condition=True)
    assert rig[3].intent.reason == "condition_already_satisfied"
    assert not any(e[0] == "down" for e in rig[1].history)


def test_expired_pickup_is_replanned_without_duplicate_button_events():
    rig = setup_drag()
    c, plant, output, r, _, _ = rig
    original = output.begin_drag
    calls = []
    def delayed(command):
        calls.append(command)
        if len(calls) == 1:
            output.events.append({"kind": "input_cancelled", "command_id": command["command_id"],
                                  "reason": "deadline_expired", "t_mono": c()})
        else:
            original(command)
    output.begin_drag = delayed
    for seq in range(2, 200):
        c.advance(.01)
        sample(rig, seq, condition=plant.cursor[0] > 190)
        if not r.intent.active:
            break
    assert r.intent.phase == "completed" and len(calls) == 2
    assert sum(e[0] == "down" for e in plant.history) == 1
    assert sum(e[0] == "up" for e in plant.history) == 1


def test_frames_started_before_release_cannot_verify_completion():
    rig = setup_drag()
    c, _, _, r, _, _ = rig
    seq = advance_to_hold(rig)
    c.advance(.01)
    sample(rig, seq, condition=True)
    released = r.intent.released_at
    c.advance(.04)
    sample(rig, seq + 1, condition=True, started=released - .001)
    assert not r.intent.condition_verified


def test_held_intent_arbitrates_point_input_and_second_program():
    from ganglion.core.schema import InputSpec
    rig = setup_drag()
    _, _, _, r, _, ids = rig
    advance_to_hold(rig)
    with pytest.raises(RuntimeErrorWithCode) as error:
        r.start_intent("agent", IntentSpec(watch_id=ids[0]))
    assert error.value.code == "pointer_busy"
    with pytest.raises(RuntimeErrorWithCode):
        r.input("agent", InputSpec(action="click", point=[10, 10], snapshot_id=r.snapshot()["id"]))


@pytest.mark.parametrize("outcome", ["arrival", "disappearance", "destination_lost"])
def test_watched_destination_and_absent_condition(outcome):
    rig = setup_drag()
    c, plant, _, r, layout, ids = rig
    r.cancel("agent", r.intent.id)
    r.tick()
    destination = r.add_watch("agent", WatchSpec(name="destination", snapshot_id=r.snapshot()["id"],
        region=layout["rect"], color_rgb=[240, 200, 40]))["watch_id"]
    r.start_intent("agent", IntentSpec(program="drag", watch_id=ids[0], destination_watch_id=destination,
        until={"watch_id": ids[0], "present": False} if outcome == "disappearance" else None))
    picked_up = False
    for seq in range(2, 220):
        c.advance(.01)
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        if not picked_up or outcome != "disappearance":
            frame[45:75, 85:115] = [120, 220, 40]
        if not picked_up or outcome != "destination_lost":
            # A watched destination changes after pickup; arrival must use its new position.
            x = 340 if picked_up else 260
            frame[45:75, x - 15:x + 15] = [40, 200, 240]
        r.observe(frame, c(), seq, layout)
        r.tick()
        picked_up |= plant.held
        if not r.intent.active:
            break
    assert picked_up and not plant.held
    if outcome == "destination_lost":
        assert r.intent.reason == "target_lost" and r.intent.phase == "failed"
    else:
        assert r.intent.phase == "completed"
        assert r.intent.condition_verified == (outcome == "disappearance")
        if outcome == "arrival":
            assert abs(plant.cursor[0] - 340) <= r.intent.spec.tolerance_px


@pytest.mark.parametrize("params", [dict(program="drag"),
    dict(program="drag", destination=[10, 10], destination_watch_id="also"),
    dict(program="drag", destination=[10, 10], click=True), dict(destination=[10, 10])])
def test_drag_schema_rejects_ambiguous_programs(params):
    with pytest.raises(ValueError):
        IntentSpec(watch_id="source", **params)
