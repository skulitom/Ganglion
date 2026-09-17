"""Align the view so a watched target sits at a screen point, then optionally fire.

This is the first-person counterpart of reach: the plant is the game camera driven by relative
mouse deltas, the feedback is the target's detected screen position, and there is no cursor to
confirm arrival with. Alignment needs distinct fresh observations inside tolerance for the
settling interval; firing is a bounded button hold; the whole program stays inside the same
lease, timeout and evidence-age rules as every other intent.

With `controller: connectome` the model proposes the view velocity each tick. The runtime keeps
a virtual cursor (the crosshair moved by every delta applied so far) and a virtual goal (the
target in that same frame), so the model sees the same goal-error and own-velocity channels as
in a cursor reach, and the same envelope accepts only proposals that close on the target.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from uuid import uuid4

from .reach import Reach, supervise

UNBOUNDED = (-10 ** 6, -10 ** 6, 2 * 10 ** 6, 2 * 10 ** 6)


@dataclass
class Align(Reach):
    stage: str = "aligning"          # aligning, firing, cooling
    shots: int = 0
    last_seen: float | None = None
    cool_until: float = 0.0
    aligned: bool = False
    applied_dx: int = 0
    applied_dy: int = 0
    consumed_observation: int = 0
    cleanup_watch: str | None = None   # a track watch created for this intent alone

    def status(self):
        return super().status() | {"stage": self.stage, "shots": self.shots, "aligned": self.aligned,
                                   "applied_delta": [self.applied_dx, self.applied_dy],
                                   "point": self.spec.point, "fire": self.spec.fire.model_dump() if self.spec.fire else None}


def align_event(runtime, intent, event, details):
    """Advance the program on helper events addressed to it."""
    if not intent.active or details.get("intent_id") != intent.id:
        return
    if event["kind"] == "input_released" and intent.stage == "firing":
        intent.shots += 1
        spec = intent.spec
        if intent.shots >= spec.fire.repeat:
            runtime._finish_intent("completed", "aligned_and_fired", stop=False)
        else:
            intent.stage = "cooling"
            intent.cool_until = runtime.clock() + spec.fire.interval_ms / 1000
            intent.settled_since = None
    elif event["kind"] == "input_cancelled" and intent.stage == "firing":
        if event.get("reason") == "deadline_expired":
            intent.stage = "aligning"
            intent.settled_since = None
        else:
            runtime._finish_intent("failed", event.get("reason", "input_cancelled"))


def _fire(runtime, intent, watch, now):
    spec = intent.spec
    cid = uuid4().hex
    runtime.pending[cid] = {"intent_id": intent.id, "watch_id": spec.watch_id,
                            "observation_id": watch.observation_id, "action": "fire"}
    try:
        runtime.output.button({"command_id": cid, "target": runtime.layout, "button": spec.fire.button,
                               "deadline": min(runtime.expires, intent.expires, watch.sample_started + .05),
                               "hold_until": min(runtime.expires, intent.expires, now + spec.fire.hold_ms / 1000)})
        intent.stage = "firing"
        intent.commands += 1
        runtime.note_self_motion(now + spec.fire.hold_ms / 1000 + 0.15)
    except Exception as exc:
        runtime.pending.pop(cid, None)
        runtime.halt(f"submit_failed: {exc}", degraded=True)


def _step(runtime, intent, watch, error, now, dt):
    """This tick's view delta in mouse counts: the model's proposal under the envelope, or the
    proportional reference. Returns (dx, dy, controller)."""
    spec = intent.spec
    ex, ey = error
    limit = spec.max_step / spec.gain          # pixels of view motion allowed per tick
    ref = (max(-limit, min(limit, ex)), max(-limit, min(limit, ey)))
    controller = "deterministic"
    step_px = ref
    if runtime.shadow is not None:
        from ganglion.brain.shadow import MotorSample
        sx, sy = intent.applied_dx / spec.gain, intent.applied_dy / spec.gain
        px, py = spec.point if spec.point is not None else runtime.client_centre()
        cursor = (px + sx, py + sy)
        goal = (cursor[0] + ex, cursor[1] + ey)
        if spec.controller == "connectome":
            velocity = runtime.shadow.proposal((intent.id, "align", runtime.layout_rev), now)
            candidate = None if velocity is None else supervise(cursor, goal, velocity, dt, spec.speed_px_s,
                                                                UNBOUNDED, spec.tolerance_px)
            if candidate is not None:
                step_px = (max(-limit, min(limit, candidate[0] - cursor[0])),
                           max(-limit, min(limit, candidate[1] - cursor[1])))
                controller = "connectome"
                intent.neural_commands += 1
            elif velocity is None:
                controller = "deterministic_stale"
                intent.stale_commands += 1
            else:
                controller = "deterministic_override"
                intent.overridden_commands += 1
        runtime.shadow.submit(MotorSample(intent.id, "align", runtime.layout_rev, watch.observation_id,
                                          watch.sample_started, now, min(runtime.expires, intent.expires),
                                          cursor, goal, (cursor[0] + ref[0], cursor[1] + ref[1]), UNBOUNDED,
                                          spec.speed_px_s, dt, flow=runtime.lptc()))
    dx = max(-spec.max_step, min(spec.max_step, round(step_px[0] * spec.gain)))
    dy = max(-spec.max_step, min(spec.max_step, round(step_px[1] * spec.gain)))
    return dx, dy, controller


def drive_align(runtime, intent, now):
    spec = intent.spec
    watch = runtime.watches.get(spec.watch_id)
    if watch is None:
        runtime._finish_intent("failed", "watch_removed")
        return
    if intent.stage == "firing":
        return
    if intent.stage == "cooling":
        if now < intent.cool_until:
            return
        intent.stage = "aligning"
    if watch.observation_id == 0:
        if now - intent.started >= 0.25:
            runtime._finish_intent("failed", "observation_stale")
        return
    if not watch.present:
        since = intent.last_seen if intent.last_seen is not None else intent.started
        if now - since > spec.absence_ms / 1000:
            runtime._finish_intent("failed" if not intent.shots else "completed",
                                   "target_lost" if not intent.shots else "target_lost_after_fire",
                                   stop=not intent.shots)
        return
    intent.last_seen = watch.sample_started
    age = now - watch.sample_started
    if age > 0.04:
        if age >= 0.25:
            intent.settled_since = None
            runtime._finish_intent("failed", "observation_stale")
        return
    if runtime.pending or now - intent.last_tick < 0.01:
        return
    dt = min(0.02, now - intent.last_tick)
    intent.last_tick = now
    detection = watch.detection
    px, py = spec.point if spec.point is not None else runtime.client_centre()
    ex, ey = detection["x"] - px, detection["y"] - py
    intent.error_px = hypot(ex, ey)
    if watch.observation_id <= intent.consumed_observation:
        return
    intent.consumed_observation = watch.observation_id
    if intent.error_px <= spec.tolerance_px:
        if intent.settled_since is None:
            intent.settled_since = watch.sample_started
        if watch.sample_started - intent.settled_since >= spec.settle_ms / 1000:
            intent.aligned = True
            if spec.fire is None:
                runtime._finish_intent("completed", "aligned", stop=False)
            else:
                _fire(runtime, intent, watch, now)
        return
    intent.settled_since = None
    dx, dy, controller = _step(runtime, intent, watch, (ex, ey), now, dt)
    intent.last_controller = controller
    if dx == 0 and dy == 0:
        return
    cid = uuid4().hex
    runtime.pending[cid] = {"intent_id": intent.id, "watch_id": spec.watch_id,
                            "observation_id": watch.observation_id, "captured_mono": watch.captured,
                            "sample_started_mono": watch.sample_started, "percept_ready_mono": now,
                            "goal": [px, py], "error_px": intent.error_px, "delta": [dx, dy], "controller": controller}
    try:
        runtime.output.look({"command_id": cid, "target": runtime.layout, "dx": dx, "dy": dy, "chunks": 1,
                             "deadline": min(runtime.expires, intent.expires, watch.sample_started + .05)})
        intent.commands += 1
        intent.applied_dx += dx
        intent.applied_dy += dy
        if watch.spec.kind == "motion":
            runtime.note_self_motion(now + .25)   # captured frames trail the turn by a few frames
    except Exception as exc:
        runtime.pending.pop(cid, None)
        runtime.halt(f"submit_failed: {exc}", degraded=True)
