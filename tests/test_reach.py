import math

import numpy as np
import pytest

from ganglion.core.output import ClickMachine, MemoryOutput
from ganglion.core.reach import step
from ganglion.core.runtime import Runtime, RuntimeErrorWithCode
from ganglion.core.schema import ArmSpec, IntentSpec, WatchSpec
from test_runtime import Clock
from test_output import Actuator, command


def setup(output_class=MemoryOutput, **params):
    clock = Clock()
    output = output_class(clock)
    runtime = Runtime(clock, output)
    layout = {"hwnd": 0, "pid": 0, "class": "simulation", "rect": [0, 0, 640, 360]}
    runtime.observe(np.zeros((360, 640, 3), dtype=np.uint8), clock(), 1, layout)
    runtime.claim("agent", 10)
    wid = runtime.add_watch("agent", WatchSpec(name="goal", snapshot_id=runtime.snapshot()["id"],
        region=layout["rect"], color_rgb=[40, 220, 120]))["watch_id"]
    runtime.start_intent("agent", IntentSpec(watch_id=wid, **params))
    return clock, output, runtime, layout, wid


def observe(rig, seq, x=420, y=180, visible=True):
    c, _, r, layout, _ = rig
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    if visible:
        frame[y - 18:y + 18, x - 18:x + 18] = [120, 220, 40]
    r.observe(frame, c(), seq, layout)
    r.tick()


def test_reach_moving_target_completes_from_distinct_feedback_samples():
    rig = setup(click=True)
    c, output, r, _, _ = rig
    for seq in range(2, 152):
        c.advance(0.01)
        observe(rig, seq, x=round(420 + 70 * math.sin((c() - 10) * 1.5)))
        if not r.intent.active:
            break
    assert r.intent.phase == "completed"
    assert r.intent.error_px <= 6
    assert sum("hold_ms" in cmd for cmd in output.commands) == 1
    assert r.intent.status()["task_success_verified"] is False
    assert r.intent.commands > 10  # actual corrections, not a one-shot teleport


class StuckOutput(MemoryOutput):
    def pointer(self, command):
        super().pointer(command)
        self.cursor = (0, 0)
        self.events[-1]["cursor"] = [0, 0]


def test_submitted_moves_do_not_prove_arrival():
    rig = setup(StuckOutput, timeout_seconds=0.3, click=True)
    c, output, r, _, _ = rig
    for seq in range(2, 45):
        c.advance(0.01)
        observe(rig, seq)
    assert r.intent.reason == "timeout"
    assert not any("hold_ms" in cmd for cmd in output.commands)


@pytest.mark.parametrize("cause", ["loss", "stale", "lease", "unwatch", "layout"])
def test_reach_stops_on_loss_staleness_expiry_or_invalidated_binding(cause):
    rig = setup()
    c, output, r, layout, wid = rig
    observe(rig, 2)
    c.advance(0.02)
    observe(rig, 3)
    if cause == "loss":
        observe(rig, 4, visible=False)
        r.tick()
        assert r.intent.reason == "target_lost"
    elif cause == "stale":
        c.advance(0.3)
        r.tick()
        assert r.intent.reason == "observation_stale"
    elif cause == "lease":
        r.expires = c()
        r.tick()
        assert r.intent.reason == "lease_expired"
    elif cause == "unwatch":
        r.remove("agent", wid, watch=True)
        assert r.intent.reason == "watch_removed"
    else:
        r.observe(r.frame, c(), 4, layout | {"rect": [1, 0, 639, 360]})
        assert r.intent.reason == "target_layout_changed"
    count = len(output.commands)
    c.advance(0.02)
    r.tick()
    assert not r.intent.active and len(output.commands) == count


def test_cancel_waits_for_helper_barrier_and_reflex_cannot_steal_pointer():
    rig = setup()
    c, output, r, _, wid = rig
    r.arm("agent", ArmSpec(watch_id=wid, trigger="present", cooldown_ms=50))
    observe(rig, 2)
    assert not any("hold_ms" in cmd for cmd in output.commands)
    old = r.intent.id
    r.cancel("agent", old)
    # Simulate an output worker that has not yet acknowledged its FIFO halt barrier.
    events, output.events = output.events, []
    with pytest.raises(RuntimeErrorWithCode, match="pending output"):
        r.start_intent("agent", IntentSpec(watch_id=wid))
    output.events = events
    r.tick()
    assert not r.pending
    r.start_intent("agent", IntentSpec(watch_id=wid))
    assert r.intent.id != old


def test_tick_polling_cannot_increase_control_rate_or_reuse_settle_sample():
    rig = setup()
    c, output, r, _, _ = rig
    observe(rig, 2)
    c.advance(0.01)
    r.tick()
    count = len(output.commands)
    for _ in range(100):
        r.status()
    assert len(output.commands) == count
    assert r.intent.active


def test_controller_caps_scheduler_stalls_and_stays_in_client_bounds():
    assert step((0, 0), (500, 0), 2, 1000, (0, 0, 640, 360)) == (20, 0)
    assert step((-500, 100), (100, 100), 0.01, 1000, (50, 50, 200, 200)) == (60, 100)


def test_helper_reports_os_cursor_even_when_sendinput_has_not_taken_effect():
    c, events = Clock(), []
    a = Actuator()
    a.cursor_pos = lambda: (3, 4)
    machine = ClickMachine(a, c, lambda *args: None, events.append)
    machine.pointer(command(c, point=(10, 20)))
    assert a.moves == [(10, 20)]
    assert events[-1]["cursor"] == [3, 4]
    machine.pointer(command(c, point=(50, 60), deadline=c() - 1))
    assert len(a.moves) == 1


def test_new_absence_cannot_trigger_a_move_from_the_previous_detection():
    rig = setup()
    c, output, r, _, _ = rig
    observe(rig, 2)
    c.advance(.02)
    count = len(output.commands)
    observe(rig, 3, visible=False)
    assert len(output.commands) == count
    assert r.intent.reason == "target_lost"


def test_renew_does_not_extend_intent_timeout_and_other_clients_cannot_cancel():
    rig = setup(timeout_seconds=.1)
    c, _, r, _, _ = rig
    original = r.intent.expires
    with pytest.raises(RuntimeErrorWithCode, match="Claim control"):
        r.cancel("other", r.intent.id)
    c.advance(.08)
    r.renew("agent", 20)
    assert r.intent.expires == original
    c.advance(.03)
    r.tick()
    assert r.intent.reason == "timeout" and r.owner == "agent"
