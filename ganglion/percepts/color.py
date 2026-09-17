"""Largest connected colour component within an agent-taught region."""
from __future__ import annotations

import numpy as np

from ganglion.core.schema import WatchSpec


def detect(frame: np.ndarray, spec: WatchSpec, *, was_present=False):
    import cv2
    x, y, w, h = spec.region
    crop = frame[y:y + h, x:x + w, :3]
    if crop.shape[:2] != (h, w):
        return None
    bgr = np.array(spec.color_rgb[::-1], dtype=np.int16)
    # A slightly wider exit band prevents threshold chatter.
    tol = min(255, spec.tolerance + (5 if was_present else 0))
    mask = cv2.inRange(crop, np.clip(bgr - tol, 0, 255).astype(np.uint8),
                       np.clip(bgr + tol, 0, 255).astype(np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < spec.min_pixels:
        return None
    # The centroid of a ring/concave component may be outside the component.
    # Choose the nearest pixel actually belonging to the selected component.
    cy, cx = np.nonzero(labels == i)
    nearest = int(np.argmin((cx - centers[i, 0]) ** 2 + (cy - centers[i, 1]) ** 2))
    return {"x": int(cx[nearest]) + x, "y": int(cy[nearest]) + y, "pixels": area,
            "bbox": [int(stats[i, 0]) + x, int(stats[i, 1]) + y,
                     int(stats[i, 2]), int(stats[i, 3])]}
