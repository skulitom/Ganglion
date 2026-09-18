"""The Half-Life pilot's pure helpers."""
from ganglion.evaluation.halflife.pilot import Pilot, align_max_step


def test_forgetting_a_watch_drops_every_reflex_armed_on_it(tmp_path):
    pilot = Pilot("endpoint", tmp_path)
    pilot.watches = {"motion": "w1", "hit": "w2"}
    pilot.reflexes = {"motion:track": "r1", "dodge": "r2", "hit:key": "r3"}
    pilot.armed = {"motion:track": {"watch_id": "w1"}, "dodge": {"watch_id": "w1"}, "hit:key": {"watch_id": "w2"}}
    pilot.armed_at = {n: 0.0 for n in pilot.armed}
    pilot._forget_reflexes("motion", "w1")
    assert set(pilot.reflexes) == {"hit:key"} and set(pilot.armed) == {"hit:key"} and set(pilot.armed_at) == {"hit:key"}
    pilot.log.close()


def test_engage_caps_the_turn_rate_unless_a_step_is_given():
    assert align_max_step({}) == 30                           # 2500 px/s in a 10 ms tick at gain 1.2
    assert align_max_step({"gain": 1.0}) == 25
    assert align_max_step({"max_turn_px_s": 6000}) == 72
    assert align_max_step({"max_step": 80, "max_turn_px_s": 100}) == 80
    assert align_max_step({"max_turn_px_s": 1}) == 1
