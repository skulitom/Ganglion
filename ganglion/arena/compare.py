"""Score a reach demo from its ledger: settling, tracking error and interventions per controller.

The reach demo already records every pointer command with the goal it aimed at, the cursor the
helper measured afterwards and which controller chose the step. This module turns those events
into the same task measures the training suite reports, so a deterministic run and a supervised
connectome run over the same fixture can be compared: successes, time to complete, time to first
come within tolerance, mean and worst tracking error, interventions (steps the envelope handed
back to the reference), stale steps (no fresh proposal) and the accepted share as a diagnostic.
"""
from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path

import numpy as np

TOLERANCE_PX = 6.0


def trial_metrics(result, tolerance=TOLERANCE_PX):
    events = result["events"]
    rows = []
    for trial in result["trials"]:
        if trial["mode"] != "reach":
            continue
        outcome = trial["outcome"]
        intent_id = outcome.get("intent_id")
        feedback = [e for e in events if e["kind"] == "pointer_feedback" and e.get("intent_id") == intent_id
                    and e.get("goal") is not None and e.get("cursor") is not None]
        errors = [hypot(e["goal"][0] - e["cursor"][0], e["goal"][1] - e["cursor"][1]) for e in feedback]
        times = [e["t_mono"] for e in feedback]
        controllers = [e.get("controller") for e in feedback]
        neural = controllers.count("connectome")
        overridden = controllers.count("deterministic_override")
        stale = controllers.count("deterministic_stale")
        considered = neural + overridden + stale
        within = next((t - times[0] for t, err in zip(times, errors) if err <= tolerance), None)
        rows.append({"id": trial["id"], "seed": trial["seed"], "completed": outcome.get("phase") == "completed",
                     "hit": trial.get("hits") == 1, "false_actions": trial.get("false_actions", 0),
                     "elapsed_seconds": trial["elapsed_seconds"], "commands": len(feedback),
                     "mean_error_px": float(np.mean(errors)) if errors else None,
                     "max_error_px": float(np.max(errors)) if errors else None,
                     "first_within_tolerance_s": within,
                     "connectome": neural, "overridden": overridden, "stale": stale,
                     "accepted_share": neural / considered if considered else None})
    return rows


def summarize(result, tolerance=TOLERANCE_PX):
    rows = trial_metrics(result, tolerance)
    considered = sum(r["connectome"] + r["overridden"] + r["stale"] for r in rows)
    within = [r["first_within_tolerance_s"] for r in rows if r["first_within_tolerance_s"] is not None]
    errors = [r["mean_error_px"] for r in rows if r["mean_error_px"] is not None]
    summary = {"controller": result.get("controller", {}).get("mode", "deterministic"),
               "environment": result.get("environment"), "session_id": result.get("session_id"),
               "trials": len(rows), "completed": sum(r["completed"] for r in rows), "hits": sum(r["hit"] for r in rows),
               "false_actions": sum(r["false_actions"] for r in rows),
               "mean_elapsed_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])) if rows else None,
               "first_within_tolerance_s": {"median": float(np.median(within)) if within else None,
                                            "max": float(np.max(within)) if within else None, "trials": len(within)},
               "tracking_error_px": {"mean": float(np.mean(errors)) if errors else None,
                                     "worst": float(np.max([r["max_error_px"] for r in rows if r["max_error_px"] is not None]))
                                     if errors else None},
               "commands": sum(r["commands"] for r in rows),
               "interventions": {"ticks": sum(r["overridden"] for r in rows),
                                 "fraction": sum(r["overridden"] for r in rows) / considered if considered else None},
               "stale": {"ticks": sum(r["stale"] for r in rows),
                         "fraction": sum(r["stale"] for r in rows) / considered if considered else None},
               "accepted_share": sum(r["connectome"] for r in rows) / considered if considered else None,
               "lost_events": result.get("lost_events"), "trials_detail": rows}
    shadow = result.get("shadow_score")
    if shadow:
        summary["inference_ms"] = shadow.get("inference_ms")
        summary["within_5ms_fraction"] = shadow.get("within_5ms_fraction")
    return summary


def compare(results):
    """results: {label: loaded reach-demo JSON}. Returns summaries and a markdown table."""
    summaries = {label: summarize(result) for label, result in results.items()}
    lines = ["| Controller | Completed | Hits | Mean time (s) | First within 6 px (median s) | Tracking error (mean px) | Interventions | Stale | Accepted |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, s in summaries.items():
        fmt = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"
        pct = lambda v: "n/a" if v is None else f"{v:.1%}"
        lines.append(f"| {label} | {s['completed']}/{s['trials']} | {s['hits']}/{s['trials']} | {fmt(s['mean_elapsed_seconds'])} | "
                     f"{fmt(s['first_within_tolerance_s']['median'])} | {fmt(s['tracking_error_px']['mean'], 1)} | "
                     f"{pct(s['interventions']['fraction'])} | {pct(s['stale']['fraction'])} | {pct(s['accepted_share'])} |")
    return {"controllers": summaries, "table": "\n".join(lines),
            "note": "same fixture and trial seeds; interventions are steps the envelope handed to the reference, "
                    "stale steps had no proposal fresh enough; the accepted share is a diagnostic, not a score"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="+", help="reach-demo JSON files, one per controller, optionally label=path")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    loaded = {}
    for item in args.results:
        label, _, given = item.rpartition("=")
        path = Path(given)
        result = json.loads(path.read_text(encoding="utf-8"))
        label = label or result.get("controller", {}).get("mode", path.stem)
        loaded[label if label not in loaded else path.stem] = result
    report = compare(loaded)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(report["table"])


if __name__ == "__main__":
    main()
