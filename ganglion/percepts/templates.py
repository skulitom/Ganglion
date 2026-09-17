"""Exact-scale matching of taught ink-mask crops. Application-independent.

Pixel-art interfaces render a glyph identically every time, so a taught crop compares by
intersection over union of ink pixels; a sliding search locates a small crop inside a region.
"""
from __future__ import annotations

import numpy as np

WHITE_FLOOR = 225


def ink_mask(bgr: np.ndarray, floor: int = WHITE_FLOOR) -> np.ndarray:
    """uint8 0/255 mask of pixels that are not near-white."""
    white = (bgr[..., 0] > floor) & (bgr[..., 1] > floor) & (bgr[..., 2] > floor)
    return np.where(white, 0, 255).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two same-sized ink masks; 1.0 for identical ink."""
    if a.shape != b.shape:
        raise ValueError("masks differ in shape")
    a, b = a > 0, b > 0
    union = int((a | b).sum())
    return 1.0 if union == 0 else int((a & b).sum()) / union


def best_label(mask: np.ndarray, templates: dict[str, np.ndarray], *, columns: int | None = None):
    """Label of the closest template by IoU, optionally comparing only the first columns."""
    best, score = None, 0.0
    for label, template in templates.items():
        a, b = (mask, template) if columns is None else (mask[:, :columns], template[:, :columns])
        value = iou(a, b)
        if value > score:
            best, score = label, value
    return best, score


def locate(mask: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Best normalised correlation of a small ink template inside a larger mask, with its offset."""
    import cv2
    if mask.shape[0] < template.shape[0] or mask.shape[1] < template.shape[1]:
        return 0.0, (0, 0)
    if not template.any():
        return 0.0, (0, 0)
    result = cv2.matchTemplate(mask.astype(np.float32) / 255, template.astype(np.float32) / 255,
                               cv2.TM_CCOEFF_NORMED)
    _, score, _, position = cv2.minMaxLoc(result)
    return float(score), (int(position[0]), int(position[1]))


def ink_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def mean_ink_colour(bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float] | None:
    pixels = bgr[mask > 0]
    if not len(pixels):
        return None
    b, g, r = pixels.reshape(-1, 3).mean(axis=0)
    return float(b), float(g), float(r)
