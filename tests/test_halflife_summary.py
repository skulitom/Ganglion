"""The pilot ledger summary: shares, outcomes, and the live flow against the applied turns."""
import json

from ganglion.evaluation.halflife.summary import load, summarise


def test_summary_counts_shares_outcomes_and_relates_flow_to_turns(tmp_path):
    t = 100.0
    rows = []
    # Ten looks of +60 counts each 10 ms apart: a steady turn to the right at 60 counts per tick.
    for i in range(10):
        rows.append({"kind": "look_done", "t_mono": t + i * .01, "delta": [60, 0],
                     "controller": "connectome" if i % 3 else "deterministic_override"})
    rows.append({"kind": "look_done", "t_mono": t + .11, "delta": [0, 0], "controller": "deterministic_stale"})
    # Model samples after the turn, each carrying the flow the runtime fed: minus the view's motion.
    for i in range(6):
        submitted = t + .12 + i * .01
        rows.append({"kind": "shadow_prediction", "t_mono": submitted, "inference_ms": 4.0 + i,
                     "sensor_input": {"submitted": submitted, "flow": [-5000.0, 0.0, 0.0, 0.0]}})
    rows.append({"kind": "reflex_fired", "t_mono": t + .2})
    rows.append({"kind": "intent_failed", "t_mono": t + .3, "program": "align", "reason": "target_lost"})
    rows.append({"kind": "intent_completed", "t_mono": t + .4, "program": "align", "reason": "aligned_and_fired"})
    rows.append({"kind": "observation_dropped", "t_mono": t + .5})
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    report = summarise(load(path), counts_per_px=1.2)
    assert report["view_commands"] == 11 and report["controller_shares"] == {"connectome": 6, "deterministic_override": 4, "deterministic_stale": 1}
    assert abs(report["connectome_share"] - 6 / 11) < 1e-9 and abs(report["stale_share"] - 1 / 11) < 1e-9
    assert report["align_outcomes"] == {"target_lost": 1, "aligned_and_fired": 1}
    assert report["reflex_fired"] == 1 and report["observations_dropped"] == 1
    assert report["inference_ms"]["p50"] == 6.5 and report["samples_with_flow"] == 6
    fit = report["flow_against_applied_turn"]
    assert fit["samples"] == 6 and fit["expected_px_s_median"] > 0 and fit["flow_px_s_median"] == 5000
    assert fit["correlation"] is None                      # a constant flow has no correlation, and no NaN
    json.dumps(report, allow_nan=False)                    # the report must serialise as the command line writes it
