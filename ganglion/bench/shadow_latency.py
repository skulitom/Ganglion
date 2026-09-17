"""Inference latency of the cursor adapter on its own, paced like the runtime, without a desktop.

The live ledgers show p50 near 5 ms but p95 at 30 ms and p99 above 45 ms for one network step.
This isolates the model: the same adapter, one sample every 10 ms as the reach loop would send
them, optionally with a busy Python thread beside it (the runtime's capture and observation
threads share the interpreter with the shadow worker). Run it in the console session and inside
the seat to tell environment from interpreter contention.
"""
from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np


def busy(stop, chunk=200):
    """Interpreter work of the kind the runtime does beside the model: array reshaping in Python."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    while not stop.is_set():
        small = frame[::4, ::4].astype(np.float32).mean()
        for _ in range(chunk):
            small = small * 1.0001
        time.sleep(0.001)


def measure(adapter, seconds, spacing, contention):
    from ganglion.brain.shadow import MotorSample
    stop, thread = threading.Event(), None
    if contention:
        thread = threading.Thread(target=busy, args=(stop,), daemon=True)
        thread.start()
    timings, steps = [], []
    adapter.reset()
    started = time.perf_counter()
    tick = 0
    try:
        while time.perf_counter() - started < seconds:
            now = time.perf_counter()
            sample = MotorSample("bench", "reach", 1, tick, now, now, now + 1,
                                 (100 + tick % 50, 100), (300, 200), (110, 100), (0, 0, 640, 360), 1200, spacing)
            t = time.perf_counter()
            result = adapter.predict(sample)
            timings.append((time.perf_counter() - t) * 1000)
            steps.append(result.get("neural_steps", 1))
            tick += 1
            wake = now + spacing
            while time.perf_counter() < wake:
                pass
    finally:
        stop.set()
        if thread is not None:
            thread.join(1)
    timings = np.array(timings)
    return {"samples": len(timings), "spacing_ms": spacing * 1000, "contention": contention,
            "inference_ms": {f"p{p}": float(np.percentile(timings, p)) for p in (50, 90, 95, 99)},
            "max_ms": float(timings.max()), "within_5ms_fraction": float((timings <= 5).mean()),
            "within_budget_fraction": float((timings <= 50).mean()),
            "neural_steps": {int(k): int((np.array(steps) == k).sum()) for k in sorted(set(steps))}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--spacing-ms", type=float, default=10)
    p.add_argument("--json")
    args = p.parse_args()
    if not 2 <= args.seconds <= 120 or not 5 <= args.spacing_ms <= 100:
        p.error("Use seconds 2–120 and spacing 5–100 ms")
    from ganglion.core.session import current
    from ganglion.brain.haltere_cursor import HaltereCursor
    adapter = HaltereCursor(args.checkpoint)
    report = {"checkpoint": args.checkpoint, "session_id": current().session_id,
              "model": {k: adapter.metadata[k] for k in ("neurons", "edges", "device", "adapter_version")},
              "runs": [measure(adapter, args.seconds, args.spacing_ms / 1000, contention)
                       for contention in (False, True)]}
    text = json.dumps(report, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
