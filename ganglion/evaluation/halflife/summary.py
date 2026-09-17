"""Summarise a pilot session's ledger: engagements, controller shares, inference, and the live
flow against the view deltas the runtime applied.

Reads the events the pilot appended to events.jsonl and prints a JSON summary. With a v4
checkpoint and --lptc-from-flow, every model sample carries the flow summary the runtime fed
it; comparing that with the look deltas applied in the preceding 60 ms says whether the live
percept measures what the training world simulated: slip = minus the view's own motion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return sorted(rows, key=lambda e: e.get("t_mono", 0))


def summarise(rows, *, counts_per_px=1.2, lag_s=.06):
    kinds = {}
    for e in rows:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    looks = [e for e in rows if e["kind"] == "look_done" and "controller" in e]
    shares = {}
    for e in looks:
        shares[e["controller"]] = shares.get(e["controller"], 0) + 1
    considered = sum(v for k, v in shares.items() if k in ("connectome", "deterministic_override", "deterministic_stale"))
    predictions = [e for e in rows if e["kind"] == "shadow_prediction"]
    ms = np.array([e["inference_ms"] for e in predictions]) if predictions else np.array([])
    intents = [e for e in rows if e["kind"] in ("intent_completed", "intent_failed") and e.get("program") == "align"]
    outcomes = {}
    for e in intents:
        outcomes[e.get("reason")] = outcomes.get(e.get("reason"), 0) + 1
    # Live flow against the applied deltas: for each sample that carried a flow summary, the
    # counts the runtime applied in the lag window before it, converted to pixels.
    fed = [e for e in predictions if e.get("sensor_input", {}).get("flow")]
    pairs = []
    look_times = np.array([e["t_mono"] for e in looks])
    look_dx = np.array([e.get("delta", [0, 0])[0] for e in looks], dtype=float)
    for e in fed:
        t = e["sensor_input"]["submitted"]
        window = (look_times > t - lag_s - .03) & (look_times <= t - .03)      # two frames of capture latency
        applied_px = look_dx[window].sum() / counts_per_px
        pairs.append((-applied_px / lag_s, e["sensor_input"]["flow"][0]))
    pairs = np.array(pairs) if pairs else np.zeros((0, 2))
    moving = pairs[np.abs(pairs[:, 0]) > 100] if len(pairs) else pairs
    fit = None
    if len(moving) >= 5:
        slope = float(np.polyfit(moving[:, 0], moving[:, 1], 1)[0])
        corr = float(np.corrcoef(moving[:, 0], moving[:, 1])[0, 1])
        fit = {"samples": int(len(moving)), "slope_flow_per_expected": slope, "correlation": corr,
               "expected_px_s_median": float(np.median(np.abs(moving[:, 0]))),
               "flow_px_s_median": float(np.median(np.abs(moving[:, 1])))}
    return {"events": len(rows), "kinds": kinds, "view_commands": len(looks), "controller_shares": shares,
            "connectome_share": shares.get("connectome", 0) / considered if considered else None,
            "stale_share": shares.get("deterministic_stale", 0) / considered if considered else None,
            "override_share": shares.get("deterministic_override", 0) / considered if considered else None,
            "align_outcomes": outcomes, "reflex_fired": kinds.get("reflex_fired", 0),
            "observations_dropped": kinds.get("observation_dropped", 0),
            "inference_ms": {f"p{p}": float(np.percentile(ms, p)) for p in (50, 95, 99)} if len(ms) else None,
            "samples_with_flow": len(fed), "flow_against_applied_turn": fit}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("events", type=Path)
    p.add_argument("--counts-per-px", type=float, default=1.2, help="the align gain: mouse counts per pixel")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    report = summarise(load(args.events), counts_per_px=args.counts_per_px)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
