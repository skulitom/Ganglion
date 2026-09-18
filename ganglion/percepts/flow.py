"""Optic flow with the view's own motion cancelled: what moves on its own while the camera moves.

The percept compares the newest frame with one a taught lag older on a reduced copy. First a
phase correlation finds the one translation that moves most of the picture (the view turning,
which can be far larger than any dense method follows), the older frame is shifted by it, and
dense flow between the aligned pair captures what remains. One global affine motion is fitted to
the total flow with a robust re-fit that ignores whatever disagrees, and the largest region whose
residual still disagrees with that model is reported. Unlike the motion watch it is not blinded
by the runtime's own commands: the ego-motion model is what those commands look like on screen.
The same fit yields wide-field summaries (translation, expansion, rotation) in the spirit of the
fly's lobula-plate tangential cells; they ride along with every detection and can feed the
connectome's lptc channel behind a flag. Application-independent.

A learned front-end would start from Flyvis (Lappalainen et al., https://github.com/TuragaLab/flyvis),
whose connectome-constrained visual system computes these motion signals from photoreceptors; this
percept is the deterministic stand-in that gives the reflex layer the same quantities today.
"""
from __future__ import annotations

from collections import deque
from math import ceil

import numpy as np

FIT_STRIDE = 2      # the affine fit uses every second row and column of the flow field
MIN_INLIERS = .5    # below this share of agreeing vectors the ego-motion summary is not credible
MAX_EXCLUDED = .75  # with more of the field given over to known movers there is too little background
EXCLUSION_MARGIN = .15   # a known mover's box is grown by this share of its size on every side


def _reduce(frame, region, scale):
    import cv2
    x, y, w, h = region
    crop = frame[y:y + h, x:x + w, :3]
    if crop.shape[:2] != (h, w):
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (max(16, w // scale), max(16, h // scale)), interpolation=cv2.INTER_AREA)


def global_shift(state, reference, current, known=None):
    """The translation that aligns most of `reference` with `current`, by phase correlation.
    `known` is a boolean mask of pixels that belong to known independent movers; they are
    windowed out so the peak comes from the background."""
    import cv2
    window = state.get("window")
    if window is None or window.shape != reference.shape:
        window = cv2.createHanningWindow(reference.shape[::-1], cv2.CV_32F)
        state["window"] = window
    if known is not None and known.any():
        window = window * (~known).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(reference.astype(np.float32), current.astype(np.float32), window)
    return float(dx), float(dy), float(response)


def known_movers(shape, region, scale, boxes):
    """A boolean (h, w) mask of the reduced field covered by the given screen boxes [x, y, w, h],
    each grown by EXCLUSION_MARGIN of its size on every side."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    rx, ry = region[0], region[1]
    for bx, by, bw, bh in boxes or ():
        gx, gy = bw * EXCLUSION_MARGIN, bh * EXCLUSION_MARGIN
        x0 = int(np.floor((bx - gx - rx) / scale))
        y0 = int(np.floor((by - gy - ry) / scale))
        x1 = int(np.ceil((bx + bw + gx - rx) / scale))
        y1 = int(np.ceil((by + bh + gy - ry) / scale))
        mask[max(0, y0):max(0, min(h, y1)), max(0, x0):max(0, min(w, x1))] = True
    return mask


def dense_flow(state, reference, current):
    """Dense flow (h, w, 2) in reduced pixels from reference to current."""
    import cv2
    engine = state.get("engine")
    if engine is None and hasattr(cv2, "DISOpticalFlow_create"):
        engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        state["engine"] = engine
    if engine is not None:
        return engine.calc(reference, current, None)
    return cv2.calcOpticalFlowFarneback(reference, current, None, 0.5, 3, 9, 2, 5, 1.1, 0)


def fit_ego_motion(flow, iterations=3, stride=FIT_STRIDE, seed=None, known=None):
    """One affine motion for the whole field, re-fitted on the vectors that agree with it.

    Returns the model (u, v) = (a0 + a1 x + a2 y, b0 + b1 x + b2 y) about the field centre, the
    fraction of sampled vectors it explains and the residual magnitude at every pixel. Translation
    is the view turning or panning, divergence is expansion (approach, forward motion), curl is roll.

    `seed` is a translation (dx, dy) the first inlier set is chosen around, normally the phase
    correlation's shift: the motion most of the picture shares. Without it the first fit is a
    plain least-squares average, which a large mover drags towards itself so that the re-fit
    never separates the two motions. `known` is a boolean (h, w) mask of pixels that belong to
    known independent movers (what a track or motion watch already follows); they never enter
    the fit, and `inlier_fraction` is the share of the remaining vectors that agree with it.
    """
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    x = (xs - (w - 1) / 2).astype(np.float32)
    y = (ys - (h - 1) / 2).astype(np.float32)
    sx, sy = x[::stride, ::stride].ravel(), y[::stride, ::stride].ravel()
    basis = np.stack((np.ones_like(sx), sx, sy), axis=1)
    u, v = flow[::stride, ::stride, 0].ravel(), flow[::stride, ::stride, 1].ravel()
    allowed = np.ones(len(sx), dtype=bool) if known is None else ~known[::stride, ::stride].ravel()
    weights = allowed.copy()
    a = b = np.zeros(3, dtype=np.float32)
    if seed is not None and allowed.any():
        # The seed's supporters, even when they are a minority of the field: the tolerance comes
        # from the closest quarter of the vectors, not from a median a large mover would own.
        sampled = np.hypot(u - seed[0], v - seed[1])
        weights = allowed & (sampled <= max(0.5, 2.5 * float(np.quantile(sampled[allowed], .25))))
    for _ in range(iterations):
        if weights.sum() < 12:
            break
        design = basis[weights]
        gram = design.T @ design
        a = np.linalg.solve(gram, design.T @ u[weights])
        b = np.linalg.solve(gram, design.T @ v[weights])
        sampled = np.hypot(u - basis @ a, v - basis @ b)
        # The tolerance is set by the current inliers' own spread, so the set can take in what
        # agrees with the fit and shed what does not without a disagreeing majority widening it.
        weights = allowed & (sampled <= max(0.5, 2.5 * float(np.median(sampled[weights]))))
    residual = np.hypot(flow[..., 0] - (a[0] + a[1] * x + a[2] * y), flow[..., 1] - (b[0] + b[1] * x + b[2] * y))
    return {"a": a.astype(float).tolist(), "b": b.astype(float).tolist(),
            "inlier_fraction": float(weights.sum() / max(1, allowed.sum())),
            "excluded_fraction": float(1 - allowed.mean()), "residual": residual}


def wide_field(model, scale, lag_seconds, *, aligned=True):
    """The lobula-plate style summary in full-resolution pixels per second and rates per second.

    `credible` is False when the phase correlation found no alignment it could trust (the view
    moved more than a third of the field over the lag, beyond what the dense flow can follow
    alone) or when fewer than half the sampled vectors agree with the fitted motion (something
    else fills the view). A consumer should treat the summary as absent then, not as zero motion
    it can rely on.
    """
    a, b = model["a"], model["b"]
    per_second = 1 / max(lag_seconds, 1e-3)
    return {"tx_px_s": a[0] * scale * per_second, "ty_px_s": b[0] * scale * per_second,
            "divergence_s": (a[1] + b[2]) * per_second, "curl_s": (b[1] - a[2]) * per_second,
            "inlier_fraction": model["inlier_fraction"], "excluded_fraction": model.get("excluded_fraction", 0.0),
            "credible": bool(aligned and model["inlier_fraction"] >= MIN_INLIERS
                             and model.get("excluded_fraction", 0.0) <= MAX_EXCLUDED)}


def detect_flow(frame: np.ndarray, spec, state: dict, *, captured: float, moving: bool, exclude=()):
    """The largest thing moving against the view's own motion, or None.

    `moving` is recorded, never used to suppress: a flow watch is meant to see through self-motion.
    `exclude` lists screen boxes [x, y, w, h] of movers the reflex layer already follows (a
    tracked target, a motion blob); they are kept out of the alignment and the ego-motion fit,
    so a target filling the view cannot pass for the view's own motion. The detection carries
    the ego-motion summary; `state["ego"]` keeps the newest one even when nothing independent
    is moving.
    """
    import cv2
    scale = spec.flow_scale
    small = _reduce(frame, spec.region, scale)
    if small is None:
        return None
    history = state.setdefault("history", deque())
    history.append((captured, small))
    if moving:
        state["during_self_motion"] = state.get("during_self_motion", 0) + 1
    lag = spec.lag_ms / 1000
    while len(history) > 1 and history[1][0] <= captured - lag:
        history.popleft()
    while len(history) > 64:
        history.popleft()
    if history[0][0] > captured - lag:
        return None
    reference_time, reference = history[0]
    h, w = small.shape
    known = known_movers((h, w), spec.region, scale, exclude) if exclude else None
    dx, dy, response = global_shift(state, reference, small, known)
    trusted = abs(dx) <= w / 3 and abs(dy) <= h / 3
    if not trusted:
        dx = dy = 0.0                     # no credible alignment: fall back to the dense field alone
    aligned = cv2.warpAffine(reference, np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32), (w, h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    flow = dense_flow(state, aligned, small)
    flow[..., 0] += np.float32(dx)
    flow[..., 1] += np.float32(dy)
    model = fit_ego_motion(flow, seed=(dx, dy), known=known)
    ego = wide_field(model, scale, captured - reference_time, aligned=trusted)
    ego["phase_response"] = response
    state["ego"] = ego
    state["frames"] = state.get("frames", 0) + 1
    residual = model["residual"]
    mask = (residual * scale > spec.flow_threshold).astype(np.uint8)
    margin = int(ceil(max(abs(dx), abs(dy)))) + 1        # the shifted border was invented by the warp
    mask[:margin, :] = mask[-margin:, :] = 0
    mask[:, :margin] = mask[:, -margin:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    x, y = spec.region[:2]
    if n >= 2:
        areas = stats[1:, cv2.CC_STAT_AREA] * (scale * scale)
        for k in np.argsort(-areas):
            i = int(k) + 1
            area = int(areas[k])
            if area < spec.min_pixels:
                break
            if spec.max_pixels is not None and area > spec.max_pixels:
                continue
            state["streak"] = state.get("streak", 0) + 1
            if state["streak"] < spec.persist:
                return None
            cx, cy = centres[i]
            own = labels == i
            # No velocity claim for the blob: dense flow inside a textureless mover is filled in
            # from its surroundings. A track watch on the blob measures how it moves.
            return {"x": int(round(cx * scale)) + x, "y": int(round(cy * scale)) + y, "pixels": area,
                    "bbox": [int(stats[i, 0]) * scale + x, int(stats[i, 1]) * scale + y,
                             int(stats[i, 2]) * scale, int(stats[i, 3]) * scale],
                    "residual_px": float(np.mean(residual[own]) * scale), "ego": ego}
    state["streak"] = 0
    return None
