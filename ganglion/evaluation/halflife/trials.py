"""Repeated engagements from one quicksave, so two core configurations can be compared on the
same fight: the same spawn, the same reflex chain, the ledger sliced per trial.

Each trial: quickload, spawn grunts from the console, back away for a moment, arm the engage
chain with the connectome in control, watch for a while, disarm. The events the pilot logged
between the trial's first and last look are its slice; `summarise` reduces a slice to the align
outcomes, the controller shares and the tracking error of the align steps.

The motion channel's condition is the flow watch: with `--flow-scale` a flow watch over the
upper view runs for the block and a core started with `--lptc-from-flow` feeds its credible
summaries to the model; without it the channel is zeros, whatever the core's flag. Alternate
blocks of both to keep the comparison inside one session.

    python -m ganglion.evaluation.halflife.trials run --dir runs/hl/pilot --label lptc-on-1 --trials 5 --flow-scale 8
    python -m ganglion.evaluation.halflife.trials run --dir runs/hl/pilot --label channel-zero-1 --trials 5
    python -m ganglion.evaluation.halflife.trials report runs/hl/trials/lptc-on-*.json runs/hl/trials/channel-zero-*.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .pilot import do
from .summary import load, summarise


def step(directory, command, timeout=30):
    r = do(directory, command, timeout)
    if not r.get("ok"):
        raise RuntimeError(f"{command.get('op')}: {r.get('error')}")
    return r["result"]


def console_open(frame_path):
    """Whether the game's console box is on screen in a saved look frame: a flat olive text area
    with grey text, a lighter title strip above it and a near-white input box below it, at the
    console's fixed place. The grave key toggles the console, so a trial must know the state
    before it types; an olive wall alone must not pass."""
    import cv2
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return False
    scale = frame.shape[1] / 1280

    def region(x0, x1, y0, y1):
        a, b, c, d = (int(round(v * scale)) for v in (x0, x1, y0, y1))
        return frame[c:d, a:b].astype(float)
    text, title, box = region(60, 500, 90, 300), region(60, 500, 44, 60), region(60, 480, 410, 424)
    if text.size == 0 or title.size == 0 or box.size == 0:
        return False
    b, g, r = text.mean(axis=(0, 1))
    spread = text.std(axis=(0, 1))
    olive = 55 <= g <= 110 and 5 <= g - b <= 30 and 0 <= g - r <= 25 and spread.max() - spread.min() < 4
    tb, tg, tr = title.mean(axis=(0, 1))
    strip = tg > g + 5 and 5 <= tg - tb <= 35 and title.std(axis=(0, 1)).max() < 25
    bright = box.mean(axis=(0, 1)).min() > 150 and box.std(axis=(0, 1)).max() < 40
    return bool(olive and strip and bright)


def console_commands(directory, lines):
    """Type console commands and leave the console closed, whatever state it was in."""
    seen = step(directory, {"op": "look"})
    if not console_open(seen["frame"]):
        step(directory, {"op": "tap", "key": "grave"})
    for text in lines:
        step(directory, {"op": "type", "text": text})
    step(directory, {"op": "tap", "key": "grave"})
    seen = step(directory, {"op": "look"})
    if console_open(seen["frame"]):
        step(directory, {"op": "tap", "key": "grave"})


def event_count(directory):
    path = directory / "events.jsonl"
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0


def run_trial(directory, *, grunts=2, back_ms=1200, watch_s=14.0, controller="connectome", max_turn_px_s=2500.0,
              speed_px_s=1200.0):
    step(directory, {"op": "tap", "key": "f7"})                     # quickload
    time.sleep(3.0)
    step(directory, {"op": "look"})
    first = event_count(directory)
    started = time.perf_counter()
    console_commands(directory, ["give monster_human_grunt"] * grunts)
    step(directory, {"op": "tap", "key": "3"})                      # the submachine gun, whatever the loadout left in hand
    if back_ms:
        step(directory, {"op": "walk", "keys": ["s"], "ms": int(back_ms)})
    step(directory, {"op": "engage", "response": "track", "controller": controller, "max_turn_px_s": max_turn_px_s,
                     "speed_px_s": speed_px_s})
    watched = step(directory, {"op": "wait", "s": watch_s}, timeout=watch_s + 10)
    step(directory, {"op": "disarm"})
    step(directory, {"op": "cancel"})
    step(directory, {"op": "look"})
    last = event_count(directory)
    return {"first_event": first, "last_event": last, "seconds": round(time.perf_counter() - started, 2),
            "watched": watched}


def acquisition(rows, intents):
    """Seconds from each firing intent's start to its first shot's release: the time the target
    took to reach the crosshair. Intents that never fired have no acquisition."""
    first_fire = {}
    for e in rows:
        if e["kind"] == "input_released" and e.get("action") == "fire" and "intent_id" in e:
            first_fire[e["intent_id"]] = min(first_fire.get(e["intent_id"], float("inf")), e["t_mono"])
    times = [first_fire[e["intent_id"]] - e["started_mono"] for e in intents
             if e.get("shots") and e.get("intent_id") in first_fire and "started_mono" in e]
    return [round(t, 3) for t in times if t >= 0]


def engagement(rows, intents):
    """Seconds from each firing intent's start to its last align step, which includes the
    re-alignment between shots and any tracking after the last one."""
    last_step = {}
    for e in rows:
        if e["kind"] == "look_done" and "intent_id" in e:
            last_step[e["intent_id"]] = max(last_step.get(e["intent_id"], 0), e["percept_ready_mono"])
    times = [last_step[e["intent_id"]] - e["started_mono"] for e in intents
             if e.get("shots") and e.get("intent_id") in last_step and "started_mono" in e]
    return [round(t, 3) for t in times if t >= 0]


def proposals(rows):
    """Where the model's align proposals point: the cosine with the goal error and, for samples
    that carried a flow, with the flow. A readout that follows the slip against the goal shows
    here before it shows in the override share."""
    to_goal, to_flow = [], []
    for e in rows:
        if e["kind"] != "shadow_prediction" or e.get("stage") != "align" or "raw_actions" not in e:
            continue
        si = e.get("sensor_input", {})
        goal = np.array(si.get("goal", [0, 0]), float) - np.array(si.get("cursor", [0, 0]), float)
        raw = np.array(e["raw_actions"][:2], float)
        if np.linalg.norm(goal) < 1 or np.linalg.norm(raw) < 1e-6:
            continue
        to_goal.append(float(raw @ goal / np.linalg.norm(raw) / np.linalg.norm(goal)))
        flow = si.get("flow")
        if flow and np.linalg.norm(flow[:2]) > 50:
            f = np.array(flow[:2], float)
            to_flow.append(float(raw @ f / np.linalg.norm(raw) / np.linalg.norm(f)))
    return {"samples": len(to_goal),
            "cosine_to_goal_mean": float(np.mean(to_goal)) if to_goal else None,
            "share_at_goal": float(np.mean(np.array(to_goal) > .5)) if to_goal else None,
            "share_away_from_goal": float(np.mean(np.array(to_goal) < 0)) if to_goal else None,
            "flow_samples": len(to_flow),
            "cosine_to_flow_mean": float(np.mean(to_flow)) if to_flow else None,
            "share_against_flow": float(np.mean(np.array(to_flow) < -.5)) if to_flow else None}


def slice_summary(rows):
    s = summarise(rows)
    looks = [e for e in rows if e["kind"] == "look_done" and "error_px" in e]
    err = np.array([e["error_px"] for e in looks]) if looks else np.array([])
    intents = [e for e in rows if e["kind"] in ("intent_completed", "intent_failed") and e.get("program") == "align"]
    fired = sum(1 for e in intents if e.get("shots"))
    acquired = acquisition(rows, intents)
    return {"view_commands": s["view_commands"], "controller_shares": s["controller_shares"],
            "acquisition_s": acquired, "engagement_s": engagement(rows, intents), "proposals": proposals(rows),
            "connectome_share": s["connectome_share"], "override_share": s["override_share"], "stale_share": s["stale_share"],
            "align_intents": len(intents), "align_outcomes": s["align_outcomes"], "intents_that_fired": fired,
            "reflex_fired": s["reflex_fired"], "samples_with_flow": s["samples_with_flow"],
            "flow_against_applied_turn": s["flow_against_applied_turn"],
            "inference_ms": s["inference_ms"], "observations_dropped": s["observations_dropped"],
            "error_px": {"mean": float(err.mean()), "median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
                         "steps": int(len(err))} if len(err) else None}


def run(args):
    directory = Path(args.dir)
    out = Path(args.out) / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.flow_scale:
        step(directory, {"op": "watch", "name": "flow", "kind": "flow", "region": [0, 0, 1280, 560],
                         "flow_scale": int(args.flow_scale)})
    else:
        step(directory, {"op": "unwatch", "name": "flow"})
    trials = []
    for i in range(args.trials):
        trial = run_trial(directory, grunts=args.grunts, back_ms=args.back_ms, watch_s=args.watch_s,
                          controller=args.controller, max_turn_px_s=args.max_turn_px_s, speed_px_s=args.speed_px_s)
        rows = load(directory / "events.jsonl", trial["first_event"], trial["last_event"])
        trial["summary"] = slice_summary(rows)
        trials.append(trial)
        print(json.dumps({"trial": i, **trial["summary"]})[:400], flush=True)
        out.write_text(json.dumps({"label": args.label, "controller": args.controller, "grunts": args.grunts,
                                   "flow_scale": args.flow_scale, "max_turn_px_s": args.max_turn_px_s,
                                   "speed_px_s": args.speed_px_s, "trials": trials}, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def pooled(report):
    """Trial-level medians and the pooled shares, so one long trial does not dominate."""
    s = [t["summary"] for t in report["trials"]]
    if not s:
        return {"trials": 0}
    shares = {}
    for x in s:
        for k, v in x["controller_shares"].items():
            shares[k] = shares.get(k, 0) + v
    total = sum(shares.values()) or 1
    outcomes = {}
    for x in s:
        for k, v in x["align_outcomes"].items():
            outcomes[k] = outcomes.get(k, 0) + v
    errors = [x["error_px"]["mean"] for x in s if x["error_px"]]
    acquired = [t for x in s for t in x.get("acquisition_s", [])]
    engaged = [t for x in s for t in x.get("engagement_s", [])]
    props = [x["proposals"] for x in s if x.get("proposals") and x["proposals"]["samples"]]
    weighted = lambda key: (float(sum(p[key] * p["samples"] for p in props if p[key] is not None)
                                  / max(1, sum(p["samples"] for p in props if p[key] is not None)))
                            if any(p[key] is not None for p in props) else None)
    flow_props = [p for p in props if p["cosine_to_flow_mean"] is not None]
    fits = [x["flow_against_applied_turn"] for x in s if x.get("flow_against_applied_turn")]
    return {"trials": len(s), "view_commands": sum(x["view_commands"] for x in s),
            "proposals": {"samples": sum(p["samples"] for p in props),
                          "cosine_to_goal_mean": weighted("cosine_to_goal_mean"),
                          "share_at_goal": weighted("share_at_goal"), "share_away_from_goal": weighted("share_away_from_goal"),
                          "flow_samples": sum(p["flow_samples"] for p in flow_props),
                          "cosine_to_flow_mean": (float(sum(p["cosine_to_flow_mean"] * p["flow_samples"] for p in flow_props)
                                                        / max(1, sum(p["flow_samples"] for p in flow_props))) if flow_props else None),
                          "share_against_flow": (float(sum(p["share_against_flow"] * p["flow_samples"] for p in flow_props)
                                                       / max(1, sum(p["flow_samples"] for p in flow_props))) if flow_props else None)},
            "flow_against_applied_turn": {"trials": len(fits),
                                          "slope_median": float(np.median([f["slope_flow_per_expected"] for f in fits])),
                                          "correlation_median": float(np.median([f["correlation"] for f in fits if f["correlation"] is not None]))
                                          if any(f["correlation"] is not None for f in fits) else None} if fits else None,
            "acquisition_s_median": float(np.median(acquired)) if acquired else None,
            "acquisition_s_p75": float(np.percentile(acquired, 75)) if acquired else None, "acquisitions": len(acquired),
            "engagement_s_median": float(np.median(engaged)) if engaged else None,
            "connectome_share": shares.get("connectome", 0) / total, "override_share": shares.get("deterministic_override", 0) / total,
            "stale_share": shares.get("deterministic_stale", 0) / total,
            "align_intents": sum(x["align_intents"] for x in s), "intents_that_fired": sum(x["intents_that_fired"] for x in s),
            "align_outcomes": outcomes, "reflex_fired": sum(x["reflex_fired"] for x in s),
            "error_px_mean_median_over_trials": float(np.median(errors)) if errors else None,
            "error_px_mean_iqr": [float(np.percentile(errors, 25)), float(np.percentile(errors, 75))] if errors else None,
            "samples_with_flow": sum(x["samples_with_flow"] for x in s),
            "inference_p95_median": float(np.median([x["inference_ms"]["p95"] for x in s if x["inference_ms"]])) if any(x["inference_ms"] for x in s) else None}


def report(args):
    out = {}
    groups = {}
    for path in args.reports:
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        out[r["label"]] = pooled(r)
        key = r["label"].rsplit("-", 1)[0]                   # lptc-on-1, lptc-on-2 -> lptc-on
        groups.setdefault(key, {"label": key, "trials": []})["trials"].extend(r["trials"])
    for key, merged in groups.items():
        if len(merged["trials"]) > max(len(json.loads(Path(p).read_text(encoding="utf-8"))["trials"]) for p in args.reports):
            out[key + " (all blocks)"] = pooled(merged)
    text = json.dumps(out, indent=2, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dir", required=True)
    r.add_argument("--out", default="runs/hl/trials")
    r.add_argument("--label", required=True)
    r.add_argument("--trials", type=int, default=5)
    r.add_argument("--grunts", type=int, default=2)
    r.add_argument("--back-ms", type=int, default=1200)
    r.add_argument("--watch-s", type=float, default=14.0)
    r.add_argument("--controller", default="connectome")
    r.add_argument("--max-turn-px-s", type=float, default=2500.0,
                   help="cap on the view motion of every align step, the reference's and the override's")
    r.add_argument("--speed-px-s", type=float, default=1200.0,
                   help="the intent speed: the model's own step limit and the scale of its goal input")
    r.add_argument("--flow-scale", type=int, default=0, help="run a flow watch at this scale for the block (0: none)")
    s = sub.add_parser("report")
    s.add_argument("reports", nargs="+")
    s.add_argument("--out")
    args = p.parse_args()
    (run if args.cmd == "run" else report)(args)


if __name__ == "__main__":
    main()
