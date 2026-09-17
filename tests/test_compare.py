"""Reach-demo ledgers become task measures: completion, settling, tracking error, interventions."""
from ganglion.arena.compare import compare, summarize, trial_metrics


def demo(controller, steps):
    """A reach-demo result with one reach trial whose feedback events follow `steps`."""
    events, t = [], 10.0
    for goal, cursor, who in steps:
        t += .01
        events.append({"kind": "pointer_feedback", "intent_id": "i1", "goal": goal, "cursor": cursor,
                       "controller": who, "t_mono": t})
    events.append({"kind": "shadow_prediction", "intent_id": "i1", "inference_ms": 3.0})
    return {"controller": {"mode": controller}, "environment": "arena", "session_id": 2, "lost_events": 0,
            "trials": [{"id": "reach-1", "seed": 1, "mode": "reach", "elapsed_seconds": .4, "hits": 1, "false_actions": 0,
                        "outcome": {"intent_id": "i1", "phase": "completed"}},
                       {"id": "periodic-1", "seed": 1, "mode": "periodic", "elapsed_seconds": 3, "hits": 0,
                        "outcome": {"attempts": 6}}],
            "events": events, "shadow_score": {"inference_ms": {"p50": 3.0, "p95": 4.0, "p99": 5.0}, "within_5ms_fraction": 1.0}}


def test_trial_metrics_count_settling_error_and_who_acted():
    steps = [([100, 100], [60, 100], "connectome"), ([100, 100], [80, 100], "deterministic_override"),
             ([100, 100], [96, 100], "deterministic_stale"), ([100, 100], [99, 100], "connectome")]
    rows = trial_metrics(demo("connectome", steps))
    assert len(rows) == 1
    row = rows[0]
    assert row["completed"] and row["hit"] and row["commands"] == 4
    assert row["connectome"] == 2 and row["overridden"] == 1 and row["stale"] == 1
    assert row["accepted_share"] == .5
    assert abs(row["first_within_tolerance_s"] - .02) < 1e-9        # third feedback event, 20 ms after the first
    assert row["mean_error_px"] == (40 + 20 + 4 + 1) / 4 and row["max_error_px"] == 40


def test_compare_tables_both_controllers_with_deterministic_having_no_interventions():
    det = demo("deterministic", [([100, 100], [60, 100], "deterministic"), ([100, 100], [100, 100], "deterministic")])
    sup = demo("connectome", [([100, 100], [70, 100], "connectome"), ([100, 100], [100, 100], "deterministic_override")])
    report = compare({"deterministic": det, "connectome": sup})
    d, s = report["controllers"]["deterministic"], report["controllers"]["connectome"]
    assert d["interventions"]["fraction"] is None and d["accepted_share"] is None
    assert s["interventions"]["fraction"] == .5 and s["accepted_share"] == .5
    assert s["inference_ms"]["p50"] == 3.0
    assert "| deterministic | 1/1 | 1/1 |" in report["table"] and "| connectome | 1/1 | 1/1 |" in report["table"]
    assert summarize(det)["tracking_error_px"]["mean"] == 20


def test_command_line_labels_files_explicitly_or_by_controller_mode(tmp_path, monkeypatch, capsys):
    import json
    from ganglion.arena.compare import main
    det = demo("deterministic", [([100, 100], [100, 100], "deterministic")])
    one = demo("connectome", [([100, 100], [100, 100], "connectome")])
    two = demo("connectome", [([100, 100], [90, 100], "deterministic_override")])
    paths = []
    for name, result in (("det", det), ("one", one), ("two", two)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(path)
    out = tmp_path / "compare.json"
    monkeypatch.setattr("sys.argv", ["compare", str(paths[0]), str(paths[1]), f"all-motor={paths[2]}", "--out", str(out)])
    main()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert list(report["controllers"]) == ["deterministic", "connectome", "all-motor"]
    assert "| all-motor | 1/1 |" in capsys.readouterr().out
