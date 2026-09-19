"""Record a region of this session's screen as timestamped JPEG frames.

Run it in the session the application lives in, beside the core: frames carry `time.monotonic()`
(the ledger's clock), so the composer can line them up with the model's inputs afterwards.

    python -m ganglion.viz.screenrec --out runs/video/fight --seconds 20 --fps 20 --region 0,0,1280,720
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def record(out, seconds, fps=20, region=None, scale=1.0, quality=88):
    import cv2
    import dxcam
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    camera = dxcam.create(output_color="BGR")
    box = None if region is None else (region[0], region[1], region[0] + region[2], region[1] + region[3])
    index, last = [], None
    end, step, due = time.monotonic() + seconds, 1 / fps, time.monotonic()
    while time.monotonic() < end:
        frame = camera.grab(region=box)                 # None while the screen has not changed
        now = time.monotonic()
        if frame is None:
            frame = last
        if frame is not None and now >= due:
            last = frame
            small = frame if scale == 1 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            name = f"{len(index):05d}.jpg"
            cv2.imwrite(str(out / name), small, [cv2.IMWRITE_JPEG_QUALITY, quality])
            index.append({"file": name, "t_mono": now})
            due = max(due + step, now)
        time.sleep(.002)
    (out / "frames.json").write_text(json.dumps({"fps": fps, "region": region, "frames": index}), encoding="utf-8")
    return len(index)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--fps", type=float, default=20)
    p.add_argument("--region", help="x,y,w,h in screen pixels (default: the whole primary display)")
    p.add_argument("--scale", type=float, default=1.0)
    a = p.parse_args()
    region = [int(v) for v in a.region.split(",")] if a.region else None
    print(f"recorded {record(a.out, a.seconds, a.fps, region, a.scale)} frames to {a.out}")


if __name__ == "__main__":
    main()
