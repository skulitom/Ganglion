"""Motion within an agent-taught region while the runtime itself is not moving the view.

The detector compares the newest frame with one a taught lag older and returns the largest
changed blob. It keeps a short frame history per watch and reports nothing while the caller
says its own commands are moving the camera or the player, clearing that history so the
first comparison after self-motion is between two still frames. Application-independent.
"""
from __future__ import annotations

from collections import deque

import numpy as np

SCALE = 2   # the diff runs on a half-resolution copy


def detect_motion(frame: np.ndarray, spec, state: dict, *, captured: float, moving: bool):
    import cv2
    x, y, w, h = spec.region
    crop = frame[y:y + h, x:x + w, :3]
    if crop.shape[:2] != (h, w):
        return None
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (max(1, w // SCALE), max(1, h // SCALE)),
                      interpolation=cv2.INTER_AREA).astype(np.float32)
    # Ratio to the local mean: a light that flashes scales a region's brightness and cancels out;
    # something that moves changes which texture sits where and does not.
    local = cv2.GaussianBlur(gray, (0, 0), 9) + 12.0
    small = np.clip(gray / local * 128.0, 0, 255).astype(np.uint8)
    history = state.setdefault("history", deque())
    if moving:
        history.clear()
        state["suppressed"] = state.get("suppressed", 0) + 1
        return None
    history.append((captured, small))
    lag = spec.lag_ms / 1000
    reference = None
    while len(history) > 1 and history[1][0] <= captured - lag:
        history.popleft()
    if history[0][0] <= captured - lag:
        reference = history[0][1]
    while len(history) > 64:
        history.popleft()
    if reference is None:
        return None
    diff = cv2.absdiff(small, reference)
    mask = (diff > spec.threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        state["streak"] = 0
        return None
    areas = stats[1:, cv2.CC_STAT_AREA] * (SCALE * SCALE)
    order = np.argsort(-areas)
    for k in order:
        i = int(k) + 1
        area = int(areas[k])
        if area < spec.min_pixels:
            break
        if spec.max_pixels is not None and area > spec.max_pixels:
            continue
        cx, cy = centers[i]
        # Persistence: the blob has to be there in consecutive comparisons before it counts.
        state["streak"] = state.get("streak", 0) + 1
        if state["streak"] < spec.persist:
            return None
        return {"x": int(round(cx * SCALE)) + x, "y": int(round(cy * SCALE)) + y, "pixels": area,
                "bbox": [int(stats[i, 0]) * SCALE + x, int(stats[i, 1]) * SCALE + y,
                         int(stats[i, 2]) * SCALE, int(stats[i, 3]) * SCALE]}
    state["streak"] = 0
    return None
