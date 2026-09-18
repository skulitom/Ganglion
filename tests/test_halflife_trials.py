"""Repeated-engagement trials: a ledger slice reduces to outcomes, shares and tracking error."""
import json

import numpy as np

from ganglion.evaluation.halflife.trials import console_open, pooled, slice_summary


def rows_for(controller_pattern, errors, outcomes, flow=None):
    t = 50.0
    rows = []
    for i, err in enumerate(errors):
        rows.append({"kind": "look_done", "t_mono": t + i * .01, "delta": [12, 0], "error_px": err, "intent_id": "i1",
                     "percept_ready_mono": t + i * .01, "controller": controller_pattern[i % len(controller_pattern)]})
        rows.append({"kind": "shadow_prediction", "t_mono": t + i * .01, "inference_ms": 3.0 + i % 4, "stage": "align",
                     "raw_actions": [1.0, 0.0, 0.0, 0.0],
                     "sensor_input": {"submitted": t + i * .01, "flow": flow, "cursor": [0, 0], "goal": [100, 0]}})
    for j, reason in enumerate(outcomes):
        rows.append({"kind": "intent_completed" if reason.startswith("aligned") else "intent_failed", "program": "align",
                     "t_mono": t + 1 + j, "reason": reason, "shots": 4 if "fire" in reason else 0,
                     "intent_id": "i1" if reason.startswith("aligned") else f"i{j + 2}", "started_mono": t - .2})
        rows.append({"kind": "reflex_fired", "t_mono": t + .5 + j})
    return rows


def test_console_box_is_recognised_in_a_look_frame(tmp_path):
    import cv2
    rng = np.random.default_rng(1)
    scene = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "scene.jpg"), scene)
    assert console_open(tmp_path / "scene.jpg") is False
    with_console = scene.copy()
    with_console[12:220, 12:300] = (63, 78, 70)                 # the console's olive box (BGR)
    text = rng.random((208, 288)) < .08                          # grey text speckle
    with_console[12:220, 12:300][text] = (120, 135, 127)
    cv2.imwrite(str(tmp_path / "console.jpg"), with_console)
    assert console_open(tmp_path / "console.jpg") is True
    assert console_open(tmp_path / "missing.jpg") is False


def test_slice_summary_reduces_a_trial_and_pooled_merges_trials():
    a = slice_summary(rows_for(["connectome", "connectome", "deterministic_override"], [80, 40, 20, 10, 5, 5],
                               ["target_lost", "aligned_and_fired"], flow=[-300.0, 0.0, 0.0, 0.0]))
    assert a["view_commands"] == 6 and a["controller_shares"] == {"connectome": 4, "deterministic_override": 2}
    assert abs(a["connectome_share"] - 4 / 6) < 1e-9 and a["stale_share"] == 0
    assert a["align_intents"] == 2 and a["intents_that_fired"] == 1 and a["align_outcomes"] == {"target_lost": 1, "aligned_and_fired": 1}
    assert a["reflex_fired"] == 2 and a["samples_with_flow"] == 6
    assert a["error_px"]["steps"] == 6 and abs(a["error_px"]["mean"] - 160 / 6) < 1e-9 and a["error_px"]["median"] == 15
    assert a["acquisition_s"] == [.25]                        # from the intent's start to its last step
    assert a["proposals"]["samples"] == 6 and a["proposals"]["cosine_to_goal_mean"] == 1.0 and a["proposals"]["share_at_goal"] == 1.0
    assert a["proposals"]["flow_samples"] == 6 and a["proposals"]["cosine_to_flow_mean"] == -1.0 and a["proposals"]["share_against_flow"] == 1.0
    b = slice_summary(rows_for(["deterministic_stale"], [200, 100], ["timeout"]))
    assert b["samples_with_flow"] == 0 and b["stale_share"] == 1 and b["intents_that_fired"] == 0
    assert b["proposals"]["flow_samples"] == 0 and b["proposals"]["cosine_to_flow_mean"] is None
    empty = slice_summary([])
    assert empty["view_commands"] == 0 and empty["error_px"] is None and empty["inference_ms"] is None
    report = {"label": "x", "trials": [{"summary": a}, {"summary": b}, {"summary": empty}]}
    p = pooled(report)
    assert p["trials"] == 3 and p["view_commands"] == 8 and p["align_intents"] == 3 and p["intents_that_fired"] == 1
    assert abs(p["connectome_share"] - 4 / 8) < 1e-9 and abs(p["stale_share"] - 2 / 8) < 1e-9
    assert p["align_outcomes"] == {"target_lost": 1, "aligned_and_fired": 1, "timeout": 1}
    assert p["error_px_mean_median_over_trials"] == (160 / 6 + 150) / 2 and p["samples_with_flow"] == 6
    assert p["acquisitions"] == 1 and p["acquisition_s_median"] == .25
    assert p["proposals"]["samples"] == 8 and p["proposals"]["cosine_to_goal_mean"] == 1.0 and p["proposals"]["flow_samples"] == 6
    assert p["proposals"]["cosine_to_flow_mean"] == -1.0
    assert pooled({"label": "none", "trials": []}) == {"trials": 0}
    json.dumps(p, allow_nan=False)
