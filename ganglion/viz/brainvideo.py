"""The fly brain on the left, the application it steers on the right.

The left panel draws every neuron of the Haltere connectome at its real position in the male CNS
(brain on top, nerve cord below), brightening as it fires. The rates are not recorded live: the
ledger holds every input the model was given (`shadow_prediction` and `shadow_discarded` events
carry the full sample, the reset flag and the step count), so the composer runs the same
checkpoint over the same inputs again and reads the rates out, then checks its proposals against
the logged ones. The right side is a screen recording made beside the core (`screenrec.py`),
on the ledger's clock.

    python -m ganglion.viz.brainvideo --events runs/hl/pilot/events.jsonl --frames runs/video/fight \\
        --checkpoint runs/cursor-dagger-v6/round-03/cursor-readout.pt --out docs/media/halflife \\
        --title "fly brain steering the view in Half-Life" --start 2.0 --seconds 10

Needs Haltere, torch with CUDA, OpenCV and Pillow, and ffmpeg on the path; none of them is a
dependency of the runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

# Haltere's population colours, with what each population does for a cursor or a view
LEGEND = (("senses (goal error in)", (0.25, 0.9, 0.9)), ("central complex", (0.95, 0.7, 0.25)),
          ("descending neurons", (0.35, 0.85, 0.45)), ("premotor", (0.55, 0.60, 0.95)),
          ("motor neurons (velocity out)", (0.98, 0.30, 0.35)))
CONTROL = {"connectome": ("fly brain", (120, 235, 140)), "deterministic_override": ("reference (overridden)", (250, 170, 80)),
           "deterministic_stale": ("reference (proposal late)", (250, 170, 80)), "deterministic": ("reference", (200, 200, 200))}


def model_events(path, start, end):
    """The ledger rows on which the model actually ran, in the order it ran them."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"shadow_' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") in ("shadow_prediction", "shadow_discarded") and "sensor_input" in e and start <= e["t_mono"] <= end:
                rows.append(e)
    return sorted(rows, key=lambda e: e.get("model_started_mono", e["t_mono"]))


def steps(path, start, end):
    """(time, controller, error in px, delta) of every pointer or view step in the span."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"controller"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") in ("look_done", "pointer_feedback") and "controller" in e and start <= e["t_mono"] <= end:
                out.append((e["t_mono"], e["controller"], e.get("error_px"), e.get("delta")))
    return sorted(out, key=lambda s: s[0])


def replay(checkpoint, events):
    """Run the checkpoint over the logged samples again: (times, rates [T, N], proposals [T, 2],
    largest difference from the logged proposals)."""
    import torch
    from ganglion.brain.haltere_cursor import HaltereCursor
    from ganglion.brain.shadow import MotorSample
    model = HaltereCursor(checkpoint, spin_sync=False)
    times, rates, actions, worst = [], [], [], 0.0
    for e in events:
        if e.get("reset") or model.previous is None:
            model.reset()
        sample = MotorSample(**{k: tuple(v) if isinstance(v, list) else v for k, v in e["sensor_input"].items()})
        out = model.predict(sample)
        v = model.state["v"]
        v = v[:, 0] if v.shape[0] != 1 else v[0]
        rates.append((model.brain.cfg.rate_max * torch.sigmoid(v)).float().cpu().numpy())
        times.append(e.get("model_started_mono", e["t_mono"]))
        actions.append(out["raw_actions"][:2])
        logged = e.get("raw_actions")
        if logged:
            worst = max(worst, abs(logged[0] - out["raw_actions"][0]), abs(logged[1] - out["raw_actions"][1]))
    return np.array(times), np.array(rates, dtype=np.float32), np.array(actions, dtype=np.float32), worst


class BrainPanel:
    """Neurons splatted into an RGB buffer at their soma positions; text with Pillow."""
    def __init__(self, layout, colors, width, height, title):
        from PIL import Image, ImageDraw, ImageFont
        self.W, self.H, self.colors = width, height, colors.astype(np.float32)
        self.Image, self.ImageDraw = Image, ImageDraw
        self.font, self.small = (self._font(ImageFont, s) for s in (max(11, height // 38), max(10, height // 46)))
        top, bottom, side = int(height * .085), int(height * .235), 10
        # the neurons are splatted at twice the size and averaged down: 30,000 of them saturate a small panel
        k = self.ss = 2
        px = (k * side + layout[:, 0] * (k * (width - 2 * side) - 1)).astype(np.int64)
        py = (k * top + (1.0 - layout[:, 1]) * (k * (height - top - bottom) - 1)).astype(np.int64)
        base = py * (k * width) + px
        offsets = np.array([dy * k * width + dx for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        self.index = (base[:, None] + offsets[None, :]).ravel()
        self.kernel = np.tile(np.array([1.0 if o == 0 else .45 for o in offsets], dtype=np.float32), (len(base), 1))
        self.bottom = height - bottom
        img = Image.new("RGB", (width, height), (0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((side, 6), title, fill=(235, 235, 235), font=self.font)
        y = self.bottom + 6
        for label, rgb in LEGEND:
            d.ellipse((side, y + 3, side + 7, y + 10), fill=tuple(int(255 * v) for v in rgb))
            d.text((side + 13, y), label, fill=(205, 205, 205), font=self.small)
            y += self.small.size + 3
        self.static = np.asarray(img).astype(np.float32)

    @staticmethod
    def _font(ImageFont, size):
        for path in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def render(self, activity, *, t, controller, error_px, proposal):
        bright = np.clip(.07 + .42 * np.clip(activity, 0, 4), 0, 1)      # dim at rest: the model runs only while an intent does
        rgb = np.clip(self.colors * (.25 + 1.1 * bright[:, None]) + .175 * np.clip(activity - 2, 0, 2)[:, None], 0, 1)
        weight = (.45 + .55 * bright)[:, None] * rgb
        k = self.ss
        buf = np.stack([np.bincount(self.index, weights=(weight[:, c][:, None] * self.kernel).ravel(), minlength=k * k * self.W * self.H)
                        for c in range(3)], axis=1)
        buf = np.clip(buf, 0, 1).reshape(self.H, k, self.W, k, 3).mean(axis=(1, 3))
        frame = np.clip(self.static + 255 * buf, 0, 255).astype(np.uint8)
        img = self.Image.fromarray(frame)
        d = self.ImageDraw.Draw(img)
        d.text((10, 8 + self.font.size), f"t = {t:5.1f} s" + ("" if error_px is None else f"   error {error_px:4.0f} px"),
               fill=(255, 255, 255), font=self.small)
        name, colour = CONTROL.get(controller, ("idle", (130, 130, 130)))
        x0, y0 = int(self.W * .60), self.bottom + 6
        d.text((x0, y0), "this step by", fill=(170, 170, 170), font=self.small)
        d.text((x0, y0 + self.small.size + 2), name, fill=colour, font=self.small)
        # the model's proposal: a velocity in units of the intent speed, drawn in a unit circle
        r = int(min(self.W * .16, (self.H - self.bottom) * .30))
        cx, cy = x0 + r + 2, self.H - r - 8
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(90, 90, 90))
        if proposal is not None:
            vx, vy = float(np.clip(proposal[0], -1.2, 1.2)), float(np.clip(proposal[1], -1.2, 1.2))
            d.line((cx, cy, cx + vx * r, cy + vy * r), fill=(250, 77, 90), width=3)
            d.ellipse((cx + vx * r - 3, cy + vy * r - 3, cx + vx * r + 3, cy + vy * r + 3), fill=(250, 77, 90))
        d.text((cx + r + 6, cy - self.small.size // 2), "proposal", fill=(170, 170, 170), font=self.small)
        return np.asarray(img)


def latest(times, t):
    """Index of the last entry at or before t, or None."""
    i = int(np.searchsorted(times, t, side="right")) - 1
    return i if i >= 0 else None


def compose(events, frames_dir, checkpoint, out, *, title, start=0.0, seconds=10.0, fps=12.5, height=400, hold=.25):
    import cv2
    from haltere.viz.render import neuron_layout
    frames_dir, out = Path(frames_dir), Path(out)
    height = height // 2 * 2                                 # yuv420p wants even sides
    index = json.loads((frames_dir / "frames.json").read_text(encoding="utf-8"))["frames"]
    frame_times = np.array([f["t_mono"] for f in index])
    t0 = frame_times[0] + start
    t1 = min(t0 + seconds, frame_times[-1])
    ran = model_events(events, t0 - 3, t1)                  # a little earlier, so the state is the live one at t0
    if not ran:
        raise SystemExit("the model did not run inside this span")
    times, rates, proposals, worst = replay(checkpoint, ran)
    print(f"replayed {len(times)} model steps; largest difference from the logged proposals {worst:.4f}")
    mu, sd = rates.mean(axis=0), np.maximum(rates.std(axis=0), .05 * max(float(rates.std()), 1e-6))
    from haltere.train.bptt import load_checkpoint
    layout, colors = neuron_layout(load_checkpoint(checkpoint, "cpu")[2])
    acted = steps(events, t0 - 1, t1)
    step_times = np.array([s[0] for s in acted])
    first = cv2.imread(str(frames_dir / index[0]["file"]))
    app_w = int(round(first.shape[1] * height / first.shape[0] / 2)) * 2
    panel = BrainPanel(layout, colors, int(height * .9) // 2 * 2, height, title)
    work = out.parent / (out.name + "-frames")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    rest = np.zeros(len(mu), dtype=np.float32)
    count = int((t1 - t0) * fps)
    for n in range(count):
        t = t0 + n / fps
        i, j, k = latest(times, t), latest(step_times, t), latest(frame_times, t)
        live = i is not None and t - times[i] <= hold      # the model runs only while an intent does
        step = acted[j] if j is not None and t - step_times[j] <= hold else None
        left = panel.render((rates[i] - mu) / sd if live else rest, t=t - t0, controller=step[1] if step else None,
                            error_px=step[2] if step else None, proposal=proposals[i] if live else None)
        app = cv2.resize(cv2.imread(str(frames_dir / index[k if k is not None else 0]["file"])), (app_w, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(work / f"{n:05d}.png"), np.hstack([left[:, :, ::-1], app]))
    encode(work, out, fps)
    shutil.rmtree(work, ignore_errors=True)
    return {"frames": count, "model_steps": len(times), "replay_error": worst}


def encode(work, out, fps):
    """An mp4 for a link and a palette gif for the page."""
    source = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(work / "%05d.png")]
    subprocess.run(source + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(out.with_suffix(".mp4"))], check=True)
    palette = work / "palette.png"
    subprocess.run(source + ["-vf", "palettegen=max_colors=160:stats_mode=diff", str(palette)], check=True)
    subprocess.run(source + ["-i", str(palette), "-lavfi", "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                             str(out.with_suffix(".gif"))], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--events", required=True, help="the ledger (events.jsonl) of the recorded session")
    p.add_argument("--frames", required=True, help="the screenrec directory")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="output path without extension")
    p.add_argument("--title", default="fly brain (male CNS connectome), live in the loop")
    p.add_argument("--start", type=float, default=0.0, help="seconds into the recording")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=12.5)
    p.add_argument("--height", type=int, default=400)
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(compose(a.events, a.frames, a.checkpoint, Path(a.out), title=a.title, start=a.start,
                             seconds=a.seconds, fps=a.fps, height=a.height)))


if __name__ == "__main__":
    main()
