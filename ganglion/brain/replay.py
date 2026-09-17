"""Replay frozen shadow observations through a model; never imports desktop input."""
import argparse
import json
from math import hypot
from pathlib import Path
import time

import numpy as np

from .shadow import MotorSample


def compare(predictor, events):
    rows = [e for e in events if e["kind"] in ("shadow_prediction", "shadow_discarded")]
    if not rows or any("sensor_input" not in e for e in rows):
        raise ValueError("Replay requires recorded shadow sensor inputs")
    if len(rows) > 5000:
        raise ValueError("Split replays larger than 5000 observations")
    timings, disagreements, errors = [], [], []
    same_points = 0
    for index, event in enumerate(rows):
        sample = MotorSample(**event["sensor_input"])
        start = time.perf_counter()
        if index == 0 or event["reset"]:
            predictor.reset()
        proposal = predictor.predict(sample)
        timings.append((time.perf_counter()-start)*1000)
        same_points += list(proposal["point"]) == event["point"]
        disagreements.append(hypot(*(proposal["point"][i]-sample.reference[i] for i in range(2))))
        if "raw_actions" in proposal and "raw_actions" in event:
            errors.append(max(abs(a-b) for a, b in zip(proposal["raw_actions"], event["raw_actions"])))
    return {"observations": len(rows), "actuation_authority": False, "promoted": False,
            "model": predictor.metadata, "same_recorded_point_fraction": same_points/len(rows),
            "max_recorded_raw_action_error": max(errors) if errors else None,
            "mean_reference_disagreement_px": float(np.mean(disagreements)),
            "inference_ms": {f"p{p}": float(np.percentile(timings, p)) for p in (50, 95, 99)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording", type=Path)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    from .haltere_cursor import HaltereCursor
    source = json.loads(args.recording.read_text())
    result = compare(HaltereCursor(args.checkpoint), source["events"])
    result["recording"] = str(args.recording)
    result["recorded_model"] = source.get("shadow", {}).get("model")
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
