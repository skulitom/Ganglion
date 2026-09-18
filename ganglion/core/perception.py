"""Watches, reflexes and the observation step: what the runtime sees and what it does at frame rate.

A watch is a taught detector over a region of the target (colour, motion, template track or optic
flow); a reflex is a bounded response armed on a watch. `observe` takes every new frame: detectors
run outside the state lock, then the results are applied under it, appearances and vanishings go
to the ledger, and armed reflexes fire once per observation within the evidence-age budget. The
runtime object owns the state; this module only reads and advances it, like the intent programs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from .aim import Align
from .errors import RuntimeErrorWithCode
from .schema import ArmSpec, IntentSpec, WatchSpec

WATCH_BUDGET = 16
EVIDENCE_AGE = 0.05          # seconds from sample start within which a detection may still act


@dataclass
class Watch:
    spec: WatchSpec
    present: bool = False
    detection: dict | None = None
    last_command: str | None = None
    captured: float = 0.0
    observation_id: int = 0
    sample_started: float = 0.0
    source: str = "provided"
    state: dict = field(default_factory=dict)


@dataclass
class Reflex:
    spec: ArmSpec
    expires: float
    fires: int = 0
    last_fire: float = float("-inf")


class ChangeSense:
    """Cheap whole-view change sense: mean absolute grey difference of the upper 60% of the
    frame at 64x36, against the previous frame. Near zero while the view is static, whatever
    the pointer or a weapon model at the bottom does."""

    def __init__(self):
        self.previous = None

    def update(self, frame):
        try:
            import cv2
            h = frame.shape[0]
            small = cv2.resize(cv2.cvtColor(frame[:int(h * 0.6), :, :3], cv2.COLOR_BGR2GRAY), (64, 36),
                               interpolation=cv2.INTER_AREA).astype("float32")
        except Exception:
            return None
        previous, self.previous = self.previous, small
        if previous is None:
            return None
        return float(abs(small - previous).mean())


def detect(watch, frame, captured, moving):
    """Run the watch's detector on a frame; pure apart from the watch's own state."""
    if watch.spec.kind == "motion":
        from ganglion.percepts.motion import detect_motion
        return detect_motion(frame, watch.spec, watch.state, captured=captured, moving=moving)
    if watch.spec.kind == "track":
        from ganglion.percepts.track import detect_track
        return detect_track(frame, watch.spec, watch.state)
    if watch.spec.kind == "flow":
        from ganglion.percepts.flow import detect_flow
        return detect_flow(frame, watch.spec, watch.state, captured=captured, moving=moving)
    from ganglion.percepts.color import detect as detect_color
    return detect_color(frame, watch.spec, was_present=watch.present)


def new_watch(runtime, spec: WatchSpec):
    """Register a watch under the lock; a track watch cuts its template from the newest frame."""
    if len(runtime.watches) >= WATCH_BUDGET:
        runtime._error("watch_budget", f"At most {WATCH_BUDGET} watches; unwatch one first.")
    watch = Watch(spec)
    if spec.kind == "track":
        from ganglion.percepts.track import init_track
        tx, ty, tw, th = spec.template_region
        x, y, w, h = runtime.layout["rect"]
        if not (x <= tx and y <= ty and tx + tw <= x + w and ty + th <= y + h):
            runtime._error("region_outside_target", "The template must lie inside the target client area.")
        try:
            watch.state = init_track(runtime.frame, spec.template_region, spec)
        except ValueError as exc:
            runtime._error("template_unusable", str(exc))
    wid = uuid4().hex
    runtime.watches[wid] = watch
    runtime.ledger.append("watch_added", runtime.clock(), critical=True, watch_id=wid, name=spec.name, watch_kind=spec.kind)
    return wid


def observe(runtime, frame, captured, seq, layout, *, source="provided", sample_started=None):
    """Run detectors outside the state lock so halt/lease checks cannot wait on vision."""
    sample_started = captured if sample_started is None else sample_started
    with runtime.lock:
        if seq <= runtime.frame_seq:
            return
        runtime.dropped_frames += max(0, seq - runtime.frame_seq - 1) if runtime.frame_seq else 0
        if layout != runtime.layout:
            if runtime.layout is not None and runtime.owner is not None:
                runtime.halt("target_layout_changed", degraded=True)
            runtime.layout, runtime.layout_rev = layout, runtime.layout_rev + 1
        runtime.frame, runtime.frame_seq, runtime.frame_time = frame, seq, captured
        runtime.change = runtime.change_sense.update(frame)
        runtime.frame_source = source
        runtime.frame_started = sample_started
        runtime.source_counts[source] = runtime.source_counts.get(source, 0) + 1
        watches = list(runtime.watches.items())
        revision = runtime.layout_rev
        moving = runtime.clock() < runtime.motion_until
    observations = [(wid, watch, detect(watch, frame, captured, moving)) for wid, watch in watches]
    with runtime.lock:
        runtime.tick(drive=False)
        if revision != runtime.layout_rev or seq != runtime.frame_seq:
            return
        if runtime.clock() - sample_started > EVIDENCE_AGE:
            runtime.dropped_frames += 1
            runtime.ledger.append("observation_dropped", runtime.clock(), observation_id=seq,
                                  reason="older_than_50_ms")
            return
        for wid, watch, detection in observations:
            if runtime.watches.get(wid) is not watch:
                continue
            appeared = detection is not None and not watch.present
            vanished = detection is None and watch.present
            watch.present, watch.detection = detection is not None, detection
            watch.captured, watch.observation_id = captured, seq
            watch.sample_started, watch.source = sample_started, source
            if watch.spec.kind == "flow":
                ego = watch.state.get("ego")
                runtime.flow = None if ego is None else dict(ego, captured_mono=captured)
            ready = runtime.clock()
            if appeared or vanished:
                runtime.ledger.append("appear" if appeared else "vanish", ready, watch_id=wid,
                                      observation_id=seq, captured_mono=captured, detection=detection)
            if vanished and watch.last_command:
                runtime.ledger.append("effect_observed", ready, critical=True, watch_id=wid,
                                      command_id=watch.last_command, observation_id=seq,
                                      condition="watched_target_absent", task_success_verified=False)
                watch.last_command = None
            if detection is None or runtime.owner is None:
                continue
            # Evidence age bounds are independent of a static desktop's capture health.
            if ready - sample_started > EVIDENCE_AGE:
                continue
            frame_facts = {"wid": wid, "seq": seq, "captured": captured, "source": source,
                           "sample_started": sample_started, "layout": layout, "ready": ready}
            for rid, reflex in list(runtime.reflexes.items()):
                if not _respond(runtime, rid, reflex, watch, detection, appeared, frame_facts):
                    break
        runtime._tick_intent(runtime.clock())


def _respond(runtime, rid, reflex, watch, detection, appeared, facts):
    """Fire one armed reflex for this observation if it applies. Returns False when the runtime
    halted on a failed submission, so the caller stops responding for this frame."""
    spec = reflex.spec
    wid, seq, ready, layout = facts["wid"], facts["seq"], facts["ready"], facts["layout"]
    if spec.watch_id != wid or (spec.trigger == "appear" and not appeared):
        return True
    if (reflex.fires >= spec.max_fires or ready >= reflex.expires
            or ready - reflex.last_fire < spec.cooldown_ms / 1000):
        return True
    if spec.response in ("click", "align", "track") and (runtime.pending or (runtime.intent and runtime.intent.active)):
        runtime.ledger.append("reflex_rejected", ready, critical=True, reflex_id=rid,
                              observation_id=seq, reason="pointer_busy")
        return True
    cid = uuid4().hex
    details = {"reflex_id": rid, "watch_id": wid, "observation_id": seq,
               "captured_mono": facts["captured"], "percept_ready_mono": ready,
               "capture_source": facts["source"], "sample_started_mono": facts["sample_started"]}
    if spec.response in ("align", "track"):
        options = spec.align.model_dump()
        target_wid, cleanup = wid, None
        if spec.response == "track":
            # Follow the thing that just moved: a template cut around its blob,
            # tracked every frame so the turn itself no longer blinds the program.
            bx, by, bw, bh = detection["bbox"]
            grow = 8
            track_spec = WatchSpec(name=f"track:{watch.spec.name}", snapshot_id=runtime.snapshot()["id"],
                                   region=layout["rect"], kind="track",
                                   template_region=(max(layout["rect"][0], bx - grow),
                                                    max(layout["rect"][1], by - grow),
                                                    min(bw + 2 * grow, 1024), min(bh + 2 * grow, 1024)))
            try:
                target_wid = cleanup = new_watch(runtime, track_spec)
            except RuntimeErrorWithCode as exc:
                runtime.ledger.append("reflex_rejected", ready, critical=True, reflex_id=rid,
                                      observation_id=seq, reason=exc.code)
                return True
        intent_spec = IntentSpec(program="align", watch_id=target_wid, **options)
        runtime.intent = Align(uuid4().hex, intent_spec, ready, ready + intent_spec.timeout_seconds)
        runtime.intent.cleanup_watch = cleanup
        runtime.state = "running"
        details["intent_id"] = runtime.intent.id
        runtime.ledger.append("intent_started", ready, critical=True, reflex_id=rid, **runtime.intent.status())
    elif spec.response == "key":
        deadline = min(runtime.expires, reflex.expires, facts["sample_started"] + EVIDENCE_AGE)
        runtime.key_pending[cid] = details
        try:
            runtime.output.key({"command_id": cid, "target": layout, "deadline": deadline,
                                "keys": [spec.key], "hold_until": min(runtime.expires, ready + spec.hold_ms / 1000)})
        except Exception as exc:
            runtime.key_pending.pop(cid, None)
            runtime.halt(f"submit_failed: {exc}", degraded=True)
            return False
        if spec.key in ("w", "a", "s", "d", "space", "ctrl", "lshift"):
            runtime.note_self_motion(ready + spec.hold_ms / 1000 + 0.35)
    if spec.response == "click":
        # The helper rechecks target identity, foreground, bounds, and this deadline.
        deadline = min(runtime.expires, reflex.expires, facts["sample_started"] + EVIDENCE_AGE)
        runtime.pending[cid] = details
        try:
            runtime.output.submit({"command_id": cid, "x": detection["x"], "y": detection["y"],
                                   "target": layout, "deadline": deadline, "hold_ms": spec.hold_ms})
        except Exception as exc:
            runtime.pending.pop(cid, None)
            runtime.halt(f"submit_failed: {exc}", degraded=True)
            return False
        watch.last_command = cid
    reflex.fires += 1
    reflex.last_fire = ready
    runtime.ledger.append("notify" if spec.response == "notify" else "reflex_fired",
                          ready, critical=True, command_id=cid, response=spec.response, **details)
    return True
