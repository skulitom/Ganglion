"""Frames for the headless reach demo, drawn again from its own record.

`ganglion reach-demo --environment synthetic` has no window to film: the target lives in memory and
the clicks go to a simulated pointer. Its JSON holds everything that happened, on one clock: the
evaluator's truth (each trial's start and seed, every click received, hits and misses) and the
core's ledger (every pointer step and every sample the model was given). The target's path is a
function of the trial's seed and time (`arena/reach_world.py`), so the scene can be drawn again
exactly. Trials alternate between clicks paced like an agent's turns, which land where the
target was, and a closed-loop reach.

    python -m ganglion.cli reach-demo --environment synthetic --trials 6 --controller connectome \\
        --shadow-checkpoint <checkpoint> --shadow-process --json runs/video/reach.json
    python -m ganglion.viz.reachclip runs/video/reach.json --out runs/video/reach
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

W, H = 640, 360


def target_at(seed, elapsed):
    """The centre of ReachWorld's target `elapsed` seconds into a trial."""
    phase = seed * 1.17
    return round(320 + 170 * math.sin(elapsed * .9 + phase)), round(180 + 75 * math.cos(elapsed * 1.1 + phase))


def timeline(result):
    """Trials as (start, end, id, seed), clicks as (t, x, y, hit), the pointer as (t, x, y)."""
    truth = sorted(result["truth"], key=lambda e: e["t_mono"])
    starts = [e for e in truth if e["kind"] == "trial_started"]
    trials = []
    for i, e in enumerate(starts):
        later = starts[i + 1]["t_mono"] if i + 1 < len(starts) else float("inf")
        hit = next((h["t_mono"] for h in truth if h["kind"] == "hit" and h.get("trial_id") == e["trial_id"]
                    and e["t_mono"] <= h["t_mono"] < later), None)
        trials.append((e["t_mono"], hit if hit is not None else min(later, e["t_mono"] + 6), e["trial_id"], e["seed"]))
    clicks = [(e["t_mono"], e["x"], e["y"], e["kind"] == "hit") for e in truth if e["kind"] in ("hit", "false_action")]
    pointer = [(e["t_mono"], *e["cursor"]) for e in result["events"] if e["kind"] == "pointer_feedback" and e.get("cursor")]
    pointer += [(t, x, y) for t, x, y, _ in clicks]
    return trials, clicks, sorted(pointer)


def draw(t, trials, clicks, pointer, scale=2):
    import cv2
    img = np.full((H * scale, W * scale, 3), (24, 20, 16), np.uint8)
    started = [tr for tr in trials if tr[0] <= t]
    trial = started[-1] if started and t <= started[-1][1] + .35 else None
    if trial is not None:
        closed = trial[2].startswith("reach")
        label = "closed-loop reach at 100 Hz, the fly brain proposing each step" if closed else "clicks paced like an agent's turns: aimed where the target was"
        cv2.putText(img, label, (14 * scale, 22 * scale), cv2.FONT_HERSHEY_SIMPLEX, .42 * scale, (225, 225, 225), max(1, scale // 2), cv2.LINE_AA)
        mine = [c for c in clicks if trial[0] <= c[0] <= min(t, trial[1] + .35)]
        hit = next((c for c in mine if c[3]), None)
        misses = sum(1 for c in mine if not c[3])
        note = (f"hit after {hit[0] - trial[0]:.2f} s" + (f", {misses} misses first" if misses else "")) if hit else (f"misses: {misses}" if misses else "")
        cv2.putText(img, note, (14 * scale, (H - 14) * scale), cv2.FONT_HERSHEY_SIMPLEX, .42 * scale,
                    (120, 235, 140) if hit else (110, 110, 250), max(1, scale // 2), cv2.LINE_AA)
        if t <= trial[1]:
            x, y = target_at(trial[3], t - trial[0])
            cv2.rectangle(img, ((x - 18) * scale, (y - 18) * scale), ((x + 18) * scale, (y + 18) * scale), (120, 220, 40), -1)
    trail = [(px, py, t - pt) for pt, px, py in pointer if 0 <= t - pt <= .5]
    for (ax, ay, age), (bx, by, _) in zip(trail, trail[1:]):
        shade = int(255 * (1 - age / .5))
        cv2.line(img, (int(ax * scale), int(ay * scale)), (int(bx * scale), int(by * scale)), (shade, shade, shade), scale, cv2.LINE_AA)
    for ct, x, y, hit in clicks:
        age = t - ct
        if 0 <= age <= .6:
            colour, radius = ((120, 235, 140) if hit else (80, 80, 250)), int((8 + 40 * age) * scale)
            cv2.circle(img, (int(x * scale), int(y * scale)), radius, colour, scale, cv2.LINE_AA)
    before = [p for p in pointer if p[0] <= t]
    if before:
        _, x, y = before[-1]
        tip = (int(x * scale), int(y * scale))
        arrow = np.array([tip, (tip[0], tip[1] + 16 * scale), (tip[0] + 4 * scale, tip[1] + 12 * scale), (tip[0] + 11 * scale, tip[1] + 11 * scale)])
        cv2.fillPoly(img, [arrow], (255, 255, 255), cv2.LINE_AA)
        cv2.polylines(img, [arrow], True, (0, 0, 0), max(1, scale // 2), cv2.LINE_AA)
    return img


def render(result_path, out, fps=25, lead=.3):
    import cv2
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trials, clicks, pointer = timeline(result)
    t, end, index = trials[0][0] - lead, trials[-1][1] + .6, []
    while t < end:
        name = f"{len(index):05d}.jpg"
        cv2.imwrite(str(out / name), draw(t, trials, clicks, pointer), [cv2.IMWRITE_JPEG_QUALITY, 92])
        index.append({"file": name, "t_mono": t})
        t += 1 / fps
    (out / "frames.json").write_text(json.dumps({"fps": fps, "region": [0, 0, W, H], "frames": index}), encoding="utf-8")
    with open(out / "events.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for e in result["events"]:
            f.write(json.dumps(e) + "\n")
    return len(index)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("result", help="the JSON written by reach-demo --environment synthetic")
    p.add_argument("--out", required=True, help="directory for the frames and the ledger")
    p.add_argument("--fps", type=float, default=25)
    a = p.parse_args()
    print(f"drew {render(a.result, a.out, a.fps)} frames to {a.out}")


if __name__ == "__main__":
    main()
