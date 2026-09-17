"""The Half-Life pilot's pure helpers."""
from ganglion.evaluation.halflife.pilot import align_max_step


def test_engage_caps_the_turn_rate_unless_a_step_is_given():
    assert align_max_step({}) == 30                           # 2500 px/s in a 10 ms tick at gain 1.2
    assert align_max_step({"gain": 1.0}) == 25
    assert align_max_step({"max_turn_px_s": 6000}) == 72
    assert align_max_step({"max_step": 80, "max_turn_px_s": 100}) == 80
    assert align_max_step({"max_turn_px_s": 1}) == 1
