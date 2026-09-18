"""Optic flow with ego-motion cancelled: independent movement is seen while the view moves."""
import numpy as np

from ganglion.core.schema import WatchSpec
from ganglion.percepts.flow import detect_flow, fit_ego_motion, known_movers, wide_field


def textured(rng, h, w):
    """A background with structure at every scale, so flow is well defined everywhere."""
    import cv2
    total = np.zeros((h, w), dtype=np.float32)
    for block, weight in ((64, .4), (32, .3), (16, .2), (8, .1)):
        layer = rng.integers(0, 256, (h // block + 2, w // block + 2)).astype(np.float32)
        total += cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC) * weight
    return np.clip(total + rng.integers(0, 20, (h, w)), 0, 255)


def scene(background, pan, obj_at, obj_size=28):
    """The background shifted by pan (a camera turn) with a bright square at obj_at on top."""
    h, w = background.shape
    px, py = pan
    shifted = np.roll(np.roll(background, py, axis=0), px, axis=1)
    frame = np.repeat(shifted[..., None], 3, axis=2).astype(np.uint8)
    x, y = obj_at
    if x >= 0 and y >= 0:                       # an object outside the frame is simply absent
        frame[y:y + obj_size, x:x + obj_size] = (40, 220, 250)
    return frame


def spec(**overrides):
    base = dict(name="f", snapshot_id="s", region=[0, 0, 320, 200], kind="flow", min_pixels=200, lag_ms=60,
                persist=1, flow_scale=2, flow_threshold=3.0)
    return WatchSpec(**(base | overrides))


def test_global_affine_fit_recovers_a_pan_and_an_expansion():
    h, w = 60, 100
    ys, xs = np.mgrid[0:h, 0:w]
    x, y = xs - (w - 1) / 2, ys - (h - 1) / 2
    pan = np.stack((np.full((h, w), 3.0), np.full((h, w), -1.5)), axis=-1).astype(np.float32)
    pan[20:30, 40:50] += (-6, 4)                                   # something moving against the pan
    model = fit_ego_motion(pan)
    assert abs(model["a"][0] - 3) < .05 and abs(model["b"][0] + 1.5) < .05
    assert model["residual"][25, 45] > 5 and model["residual"][5, 5] < .1
    summary = wide_field(model, scale=4, lag_seconds=.06)
    assert abs(summary["tx_px_s"] - 200) < 5 and abs(summary["ty_px_s"] + 100) < 5
    zoom = np.stack((0.02 * x, 0.02 * y), axis=-1).astype(np.float32)
    expansion = wide_field(fit_ego_motion(zoom), scale=1, lag_seconds=1)
    assert abs(expansion["divergence_s"] - .04) < 1e-3 and abs(expansion["curl_s"]) < 1e-3


def test_independent_motion_is_found_while_the_view_pans_and_not_when_only_the_view_moves():
    rng = np.random.default_rng(3)
    background = textured(rng, 200, 320)
    state, s = {}, spec()
    # Camera pans 6 px per frame; a square walks the other way. Frames 15 ms apart, lag 60 ms.
    detections = []
    for i in range(8):
        pan = (6 * i, 0)
        frame = scene(background, pan, (200 - 4 * i, 80))
        detections.append(detect_flow(frame, s, state, captured=i * .015, moving=True))
    assert detections[:4] == [None] * 4                       # no reference a full lag older yet
    found = [d for d in detections[4:] if d is not None]
    assert found, "the walking square was not seen through the pan"
    hit = found[-1]
    assert abs(hit["x"] - (200 - 4 * 7 + 14)) < 12 and abs(hit["y"] - 94) < 12
    assert hit["ego"]["tx_px_s"] > 250                        # the pan itself: 6 px per 15 ms = 400 px/s
    assert hit["residual_px"] > 3 and hit["pixels"] >= 200   # its flow disagrees with the pan
    assert state["during_self_motion"] == 8                   # recorded, not suppressed
    # The same pan with nothing else moving: ego-motion only, no detection, summary still kept.
    state2 = {}
    quiet = [detect_flow(scene(background, (6 * i, 0), (-100, -100)), s, state2, captured=i * .015, moving=True)
             for i in range(8)]
    assert all(d is None for d in quiet)
    assert state2["ego"]["inlier_fraction"] > .9 and state2["ego"]["tx_px_s"] > 250
    assert state2["ego"]["credible"]                          # a pan well inside the percept's range


def test_the_summary_is_not_credible_beyond_the_alignment_range_or_without_agreement():
    rng = np.random.default_rng(5)
    background = textured(rng, 200, 320)
    state, s = {}, spec()
    # 30 px per 15 ms frame is 120 px over the 60 ms lag, more than a third of the 320 px field:
    # the phase correlation's shift is discarded and the dense flow alone cannot follow it.
    for i in range(8):
        detect_flow(scene(background, (30 * i, 0), (-100, -100)), s, state, captured=i * .015, moving=True)
    assert state["ego"]["credible"] is False
    # The fit alone, on a field the view's motion (3, 0) shares with a mover at (-12, 6).
    # Seeded with the alignment, the fit keeps the background and reports how much agreed;
    # unseeded, the least-squares start sits between the two and never separates them.
    field = np.full((60, 100, 2), (3.0, 0.0), dtype=np.float32) + rng.normal(0, .3, (60, 100, 2)).astype(np.float32)
    field[:, :45] = (-12.0, 6.0)                              # a mover over 45 % of the field
    seeded = fit_ego_motion(field, seed=(3.0, 0.0))
    assert abs(seeded["a"][0] - 3) < .1 and abs(seeded["b"][0]) < .1 and .5 < seeded["inlier_fraction"] < .6
    assert wide_field(seeded, scale=2, lag_seconds=.06)["credible"] is True
    unseeded = fit_ego_motion(field)
    assert abs(unseeded["a"][0] - 3) > 2                     # dragged by the mover
    field[:, :60] = (-12.0, 6.0)                              # the mover fills 60 %: the alignment is the minority
    minority = fit_ego_motion(field, seed=(3.0, 0.0))
    assert minority["inlier_fraction"] < .5
    assert wide_field(minority, scale=2, lag_seconds=.06)["credible"] is False
    steady = np.full((60, 100, 2), (3.0, 0.0), dtype=np.float32)
    assert wide_field(fit_ego_motion(steady), scale=2, lag_seconds=.06)["credible"] is True
    assert wide_field(fit_ego_motion(steady), scale=2, lag_seconds=.06, aligned=False)["credible"] is False


def test_known_movers_are_kept_out_of_the_ego_motion_even_when_they_fill_the_view():
    rng = np.random.default_rng(6)
    background = textured(rng, 200, 320)
    s = spec()
    # A textured mover covering 45 % of the field walks left while the view pans right: with its
    # box excluded the alignment and the fit follow the pan on the background alone.
    mover = textured(rng, 200, 144)
    def frame_at(i):
        f = scene(background, (6 * i, 0), (-100, -100))
        x0 = 100 - 3 * i
        f[:, x0:x0 + 144] = np.repeat(mover[..., None], 3, axis=2).astype(np.uint8)
        return f, [x0, 0, 144, 200]
    plain, excluded = {}, {}
    for i in range(8):
        f, box = frame_at(i)
        detect_flow(f, s, plain, captured=i * .015, moving=True)
        detect_flow(f, s, excluded, captured=i * .015, moving=True, exclude=[box])
    assert plain["ego"]["inlier_fraction"] < .8                              # the mover sits in the fit's field
    ego = excluded["ego"]
    assert ego["tx_px_s"] > 250 and abs(ego["ty_px_s"]) < 60 and ego["credible"]   # the pan: 6 px per 15 ms = 400 px/s
    assert .45 <= ego["excluded_fraction"] <= .75 and ego["inlier_fraction"] > .6
    # The mask itself: screen boxes to the reduced grid, grown by a margin, clipped to the field.
    mask = known_movers((100, 160), [0, 0, 320, 200], 2, [[40, 20, 80, 40]])
    assert mask[10:30, 20:60].all() and not mask[:, 80:].any() and not mask[40:, :].any()
    assert not known_movers((100, 160), [0, 0, 320, 200], 2, []).any()
    # Too little background left: not credible even when what remains agrees.
    field = np.full((60, 100, 2), (3.0, 0.0), dtype=np.float32)
    known = np.zeros((60, 100), dtype=bool)
    known[:, :80] = True
    model = fit_ego_motion(field, seed=(3.0, 0.0), known=known)
    assert abs(model["a"][0] - 3) < .05 and model["inlier_fraction"] > .9 and model["excluded_fraction"] > .75
    assert wide_field(model, scale=2, lag_seconds=.06)["credible"] is False


def test_persistence_and_size_limits_apply_and_a_missing_region_is_ignored():
    rng = np.random.default_rng(4)
    background = textured(rng, 200, 320)
    s = spec(persist=3, max_pixels=100000)
    state = {}
    seen = []
    for i in range(10):
        frame = scene(background, (0, 0), (100 + 5 * i, 60))
        seen.append(detect_flow(frame, s, state, captured=i * .015, moving=False) is not None)
    assert seen[:6] == [False] * 6 and any(seen[6:])           # lag, then two comparisons of persistence
    tiny = spec(min_pixels=100000)
    state = {}
    assert all(detect_flow(scene(background, (0, 0), (100 + 5 * i, 60)), tiny, state, captured=i * .015, moving=False) is None
               for i in range(10))
    outside = spec(region=[0, 0, 640, 400])
    assert detect_flow(scene(background, (0, 0), (10, 10)), outside, {}, captured=0, moving=False) is None
