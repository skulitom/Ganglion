import numpy as np
import pytest

from ganglion.core.ledger import Ledger
from ganglion.core.output import MemoryOutput
from ganglion.core.runtime import Runtime, RuntimeErrorWithCode
from ganglion.core.schema import ArmSpec, WatchSpec
from ganglion.percepts.color import detect


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def rig():
    clock = Clock()
    output = MemoryOutput(clock)
    runtime = Runtime(clock, output)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    target = {"hwnd": 7, "pid": 42, "class": "test", "rect": [0, 0, 120, 100]}
    runtime.observe(frame, clock(), 1, target)
    runtime.claim("agent", 1)
    spec = WatchSpec(name="green", snapshot_id=runtime.snapshot()["id"],
                     region=[0, 0, 120, 100], color_rgb=[40, 220, 120])
    wid = runtime.add_watch("agent", spec)["watch_id"]
    return clock, output, runtime, frame, target, wid


def target_frame(frame):
    frame = frame.copy()
    frame[20:40, 30:50] = [120, 220, 40]
    return frame


def test_appearance_fires_once_per_observation_and_per_transition(rig):
    c, output, r, frame, target, wid = rig
    r.arm("agent", ArmSpec(watch_id=wid, cooldown_ms=50))
    green = target_frame(frame)
    r.observe(green, c(), 2, target)
    r.tick()
    r.observe(green, c(), 2, target)  # duplicate
    c.advance(0.1)
    r.observe(green, c(), 3, target)  # held presence is not another appearance
    assert len(output.commands) == 1
    r.observe(frame, c(), 4, target)
    r.observe(green, c(), 5, target)
    assert len(output.commands) == 2
    kinds = [e["kind"] for e in r.ledger.read(limit=200)["events"]]
    assert "effect_observed" in kinds


def test_expiry_releases_and_read_only_status_does_not_renew(rig):
    c, output, r, frame, target, wid = rig
    r.arm("agent", ArmSpec(watch_id=wid))
    original = r.expires
    c.advance(0.8)
    r.status()
    assert r.expires == original
    c.advance(0.3)
    r.observe(target_frame(frame), c(), 2, target)
    assert output.commands == []
    assert r.state == "halted" and r.reason == "lease_expired"
    with pytest.raises(RuntimeErrorWithCode, match="Claim control"):
        r.renew("agent", 1)


def test_other_client_cannot_claim_renew_or_mutate(rig):
    _, _, r, _, _, wid = rig
    for op in (lambda: r.claim("other", 1), lambda: r.renew("other", 1),
               lambda: r.arm("other", ArmSpec(watch_id=wid))):
        with pytest.raises(RuntimeErrorWithCode):
            op()


def test_level_trigger_cooldown_budget_and_reflex_ttl(rig):
    c, output, r, frame, target, wid = rig
    r.arm("agent", ArmSpec(watch_id=wid, trigger="present", cooldown_ms=100, max_fires=2))
    for seq in range(2, 12):
        r.observe(target_frame(frame), c(), seq, target)
        c.advance(0.03)
    r.tick()
    assert len(output.commands) == 2
    r.reflexes.clear()
    r.arm("agent", ArmSpec(watch_id=wid, trigger="present", ttl_seconds=0.1))
    c.advance(0.11)
    r.observe(target_frame(frame), c(), 12, target)
    assert len(output.commands) == 2


def test_layout_change_invalidates_watches_and_old_snapshot(rig):
    c, output, r, frame, target, wid = rig
    old_spec = r.watches[wid].spec
    r.arm("agent", ArmSpec(watch_id=wid))
    changed = target | {"rect": [5, 0, 115, 100]}
    r.observe(target_frame(frame), c(), 2, changed)
    assert r.state == "degraded" and not output.commands
    r.claim("agent", 1)
    with pytest.raises(RuntimeErrorWithCode, match="rebind"):
        r.add_watch("agent", old_spec)


def test_stale_observation_never_clicks(rig):
    c, output, r, frame, target, wid = rig
    r.arm("agent", ArmSpec(watch_id=wid))
    r.observe(target_frame(frame), c() - 0.2, 2, target)
    assert not output.commands
    r.observe(target_frame(frame), c(), 3, target)
    assert len(output.commands) == 1  # dropping old evidence must not consume the appearance edge


def test_competing_reflexes_do_not_share_pointer(rig):
    c, output, r, frame, target, wid = rig
    r.arm("agent", ArmSpec(watch_id=wid))
    r.arm("agent", ArmSpec(watch_id=wid))
    r.observe(target_frame(frame), c(), 2, target)
    assert len(output.commands) == 1
    assert any(e["kind"] == "reflex_rejected" for e in r.ledger.read()["events"])


def test_worker_failure_is_visible(rig):
    _, output, r, _, _, _ = rig
    output.alive = False
    assert r.status()["reason"] == "input_worker_lost"


def test_ledger_gaps_pagination_and_readers():
    ledger = Ledger(perception_capacity=2, action_capacity=2)
    for i in range(6):
        ledger.append("motion", i)
    ledger.append("input_failed", 7, critical=True)
    first = ledger.read(limit=1)
    assert first["lost_events"] == 4 and first["has_more"]
    assert ledger.read(limit=1) == first  # another reader has an independent cursor
    second = ledger.read(first["next_cursor"], limit=2)
    assert [e["seq"] for e in second["events"]] == [6, 7]
    assert second["lost_events"] == 0 and not second["has_more"]
    with pytest.raises(ValueError, match="another core"):
        ledger.read("other:0")


def test_color_detector_selects_real_pixel_not_empty_centroid(rig):
    _, _, r, frame, _, wid = rig
    frame[20:60, 20:60] = [120, 220, 40]
    frame[25:55, 25:55] = 0
    result = detect(frame, r.watches[wid].spec)
    assert frame[result["y"], result["x"]].tolist() == [120, 220, 40]


def test_lptc_feed_hands_over_only_a_credible_flow_summary():
    clock = Clock()
    runtime = Runtime(clock, MemoryOutput(clock), lptc_feed=True)
    assert runtime.lptc() is None                             # no flow watch has reported yet
    runtime.flow = {"tx_px_s": -800.0, "ty_px_s": 5.0, "divergence_s": .1, "curl_s": -.2,
                    "inlier_fraction": .95, "credible": True}
    assert runtime.lptc() == (-800.0, 5.0, .1, -.2)
    runtime.flow["credible"] = False                          # the view outran the percept, or the scene disagreed
    assert runtime.lptc() is None
    runtime.flow["credible"] = True
    runtime.flow["captured_mono"] = clock()
    assert runtime.lptc() is not None
    clock.advance(.2)                                         # the watch is gone: its last reading must not linger
    assert runtime.lptc() is None
    quiet = Runtime(clock, MemoryOutput(clock))
    quiet.flow = dict(runtime.flow, credible=True)
    assert quiet.lptc() is None                               # the feed is off by default


def test_input_failure_does_not_clear_failed_release_tracking():
    from ganglion.core.actuators import Actuators
    a = Actuators()
    class Blocked:
        def SendInput(self, *args):
            return 0
    a.user32 = Blocked()
    a.held_buttons.add("left")
    with pytest.raises(OSError, match="0/1"):
        a.release_all()
    assert a.held_buttons == {"left"}
