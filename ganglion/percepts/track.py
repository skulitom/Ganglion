"""Track a taught crop with normalised cross-correlation: position feedback while the view moves.

A track watch starts from a crop of the newest frame (an agent region, or the blob another watch
just detected). Every frame it searches a window around the last position for the template,
updates the position when the match is good enough, and slowly blends the template toward the
new appearance. A run of misses reports the target absent; the watch keeps its last position so
a target that reappears nearby is picked up again. Application-independent.
"""
from __future__ import annotations

import numpy as np

MIN_SIDE = 32
MAX_MISSES = 8


def _gray(frame):
    import cv2
    return cv2.cvtColor(frame[..., :3], cv2.COLOR_BGR2GRAY)


def init_track(frame: np.ndarray, bbox, spec) -> dict:
    """Cut the template from bbox (x, y, w, h), expanded to at least MIN_SIDE and clamped."""
    h_frame, w_frame = frame.shape[:2]
    x, y, w, h = (int(v) for v in bbox)
    if w < MIN_SIDE:
        x, w = x - (MIN_SIDE - w) // 2, MIN_SIDE
    if h < MIN_SIDE:
        y, h = y - (MIN_SIDE - h) // 2, MIN_SIDE
    x, y = max(0, min(x, w_frame - w)), max(0, min(y, h_frame - h))
    w, h = min(w, w_frame - x), min(h, h_frame - y)
    if w < 8 or h < 8:
        raise ValueError("template too small")
    gray = _gray(frame)
    template = gray[y:y + h, x:x + w].astype(np.float32)
    if float(template.std()) < 2.0:
        raise ValueError("template has no texture to track")
    return {"template": template, "x": x + w // 2, "y": y + h // 2, "w": w, "h": h,
            "score": 1.0, "misses": 0, "frames": 0}


def detect_track(frame: np.ndarray, spec, state: dict):
    import cv2
    if "template" not in state:
        return None
    gray = _gray(frame)
    rx, ry, rw, rh = spec.region
    tw, th = state["w"], state["h"]
    search = spec.search_px
    x0 = max(rx, state["x"] - tw // 2 - search)
    y0 = max(ry, state["y"] - th // 2 - search)
    x1 = min(rx + rw, state["x"] + tw // 2 + search)
    y1 = min(ry + rh, state["y"] + th // 2 + search)
    state["frames"] += 1
    if x1 - x0 < tw or y1 - y0 < th:
        state["misses"] += 1
        return None
    window = gray[y0:y1, x0:x1].astype(np.float32)
    result = cv2.matchTemplate(window, state["template"], cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    if not np.isfinite(score) or score < spec.min_score:
        state["misses"] += 1
        state["score"] = float(score) if np.isfinite(score) else 0.0
        if state["misses"] > MAX_MISSES:
            state.pop("template", None)   # lost for good: the watch stays absent
        return None
    lx, ly = int(location[0]) + x0, int(location[1]) + y0
    state["x"], state["y"], state["score"], state["misses"] = lx + tw // 2, ly + th // 2, float(score), 0
    if spec.update > 0:
        fresh = gray[ly:ly + th, lx:lx + tw].astype(np.float32)
        if fresh.shape == state["template"].shape:
            state["template"] = (1 - spec.update) * state["template"] + spec.update * fresh
    return {"x": state["x"], "y": state["y"], "pixels": tw * th, "bbox": [lx, ly, tw, th], "score": round(float(score), 3)}
