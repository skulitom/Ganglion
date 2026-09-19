"""The video tools' pure parts: the redrawn reach scene, the ledger readers and the brain panel."""
import json

import numpy as np
import pytest

from ganglion.arena.reach_world import ReachWorld
from ganglion.viz.brainvideo import latest, model_events, steps
from ganglion.viz.reachclip import draw, render, target_at, timeline


def test_the_redrawn_target_is_the_worlds_own():
    now = [100.0]
    world = ReachWorld(clock=lambda: now[0])
    world.reset({"id": "reach-3", "seed": 3})
    for elapsed in (0.0, .37, 1.9, 4.2):
        now[0] = 100.0 + elapsed
        world.update()
        x, y = target_at(3, elapsed)
        assert world.box == (x - 18, y - 18, 36, 36)


def result():
    truth = [{"kind": "trial_started", "t_mono": 10.0, "trial_id": "periodic-1", "seed": 1},
             {"kind": "false_action", "t_mono": 10.5, "trial_id": "periodic-1", "x": 400, "y": 120},
             {"kind": "trial_started", "t_mono": 13.0, "trial_id": "reach-1", "seed": 1},
             {"kind": "hit", "t_mono": 13.6, "trial_id": "reach-1", "x": 480, "y": 160}]
    events = [{"kind": "pointer_feedback", "t_mono": 13.2, "cursor": [200, 110], "controller": "connectome", "error_px": 280.0},
              {"kind": "pointer_feedback", "t_mono": 13.4, "cursor": [350, 140], "controller": "deterministic_override"},
              {"kind": "shadow_discarded", "t_mono": 13.25, "model_started_mono": 13.21, "sensor_input": {"cursor": [200, 110]}},
              {"kind": "shadow_prediction", "t_mono": 13.22, "model_started_mono": 13.19, "sensor_input": {"cursor": [190, 110]}},
              {"kind": "shadow_prediction", "t_mono": 13.3},                      # no sample: the model did not run on it
              {"kind": "intent_completed", "t_mono": 13.6}]
    return {"truth": truth, "events": events}


def test_the_timeline_pairs_trials_with_their_hits_and_draws_them(tmp_path):
    trials, clicks, pointer = timeline(result())
    assert trials == [(10.0, 13.0, "periodic-1", 1), (13.0, 13.6, "reach-1", 1)]          # a missed trial ends when the next starts
    assert clicks == [(10.5, 400, 120, False), (13.6, 480, 160, True)]
    assert [p[0] for p in pointer] == [10.5, 13.2, 13.4, 13.6]
    during, after = draw(13.3, trials, clicks, pointer, scale=1), draw(15.0, trials, clicks, pointer, scale=1)
    x, y = target_at(1, .3)
    assert tuple(during[y, x]) == (120, 220, 40) and tuple(after[y, x]) == (24, 20, 16)      # the target, then the empty table
    (tmp_path / "r.json").write_text(json.dumps(result()), encoding="utf-8")
    count = render(tmp_path / "r.json", tmp_path / "frames", fps=5)
    index = json.loads((tmp_path / "frames" / "frames.json").read_text(encoding="utf-8"))["frames"]
    assert count == len(index) and index[0]["t_mono"] == pytest.approx(9.7) and (tmp_path / "frames" / index[-1]["file"]).exists()
    assert len((tmp_path / "frames" / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_the_ledger_readers_keep_what_the_model_ran_on_in_the_order_it_ran(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in result()["events"]) + "\nnot json\n", encoding="utf-8")
    ran = model_events(path, 13.0, 14.0)
    assert [e["kind"] for e in ran] == ["shadow_prediction", "shadow_discarded"]           # by the time the model started on them
    assert model_events(path, 13.23, 14.0)[0]["kind"] == "shadow_discarded"
    assert steps(path, 13.0, 14.0) == [(13.2, "connectome", 280.0, None), (13.4, "deterministic_override", None, None)]
    times = np.array([1.0, 2.0, 3.0])
    assert latest(times, .5) is None and latest(times, 2.0) == 1 and latest(times, 9.0) == 2


def test_the_brain_panel_is_dim_at_rest_and_brighter_where_neurons_fire():
    pytest.importorskip("PIL")
    from ganglion.viz.brainvideo import BrainPanel
    rng = np.random.default_rng(0)
    layout, colors = rng.random((500, 2)), np.tile([.3, .9, .9], (500, 1))
    panel = BrainPanel(layout, colors, 180, 200, "test")
    rest = panel.render(np.zeros(500, np.float32), t=0.0, controller=None, error_px=None, proposal=None)
    firing = panel.render(np.full(500, 3.0, np.float32), t=1.0, controller="connectome", error_px=42.0, proposal=(.9, -.2))
    assert rest.shape == firing.shape == (200, 180, 3) and rest.dtype == np.uint8
    body = slice(20, 150)
    assert firing[body].mean() > 2 * rest[body].mean() > 0
