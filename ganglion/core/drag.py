"""Reach, hold, move, release, and verify a taught visual condition."""
from dataclasses import dataclass
from math import hypot
from uuid import uuid4

from .reach import Reach


@dataclass
class Drag(Reach):
    stage: str = "approaching"
    held: bool = False
    released_at: float = 0.0
    verified_since: float | None = None
    last_verification: int = 0
    condition_verified: bool = False
    condition_sample_started: float = 0.0

    def status(self):
        return super().status() | {"stage": self.stage, "button_held": self.held,
            "condition_verified": self.condition_verified,
            "released_mono": self.released_at, "verification_observation_id": self.last_verification,
            "condition_sample_started_mono": self.condition_sample_started,
            "condition_stable_since_mono": self.verified_since,
            "destination": self.spec.destination, "destination_watch_id": self.spec.destination_watch_id,
            "until": self.spec.until.model_dump() if self.spec.until else None}


def drag_event(runtime, intent, event):
    kind = event["kind"]
    if kind == "drag_started":
        intent.held = True
        if intent.active:
            intent.stage = "dragging"
            intent.cursor, intent.cursor_time = tuple(event["cursor"]), event["t_mono"]
            intent.settled_since = None
    elif kind == "input_released":
        intent.held = False
        if not intent.active:
            return
        if intent.stage != "releasing" or event.get("reason") != "requested":
            runtime._finish_intent("failed", event.get("reason", "unexpected_release"))
            return
        intent.stage, intent.released_at = "verifying", event["t_mono"]
        intent.verified_since = None
        if intent.spec.until is None:
            runtime._finish_intent("completed", "cursor_confirmed_drag_released", stop=False)


def _fresh(runtime, intent, watches, now):
    if any(w is None for w in watches):
        runtime._finish_intent("failed", "watch_removed")
        return False
    age = max((now - w.sample_started if w.observation_id else now - intent.started for w in watches), default=0)
    if any(not w.observation_id for w in watches) or age > .04:
        if age >= .25:
            intent.settled_since = intent.verified_since = None
            runtime._finish_intent("failed", "observation_stale")
        return False
    return True


def _arrived(intent, goal, now):
    if intent.cursor is None or now - intent.cursor_time > .05:
        intent.settled_since = None
        return False
    intent.error_px = hypot(goal[0] - intent.cursor[0], goal[1] - intent.cursor[1])
    if intent.cursor_time <= intent.consumed_cursor_time:
        return False
    intent.consumed_cursor_time = intent.cursor_time
    if intent.error_px > intent.spec.tolerance_px:
        intent.settled_since = None
        return False
    if intent.settled_since is None:
        intent.settled_since = intent.cursor_time
    return intent.cursor_time - intent.settled_since >= intent.spec.settle_ms / 1000


def drive_drag(runtime, intent, now):
    spec = intent.spec
    condition = runtime.watches.get(spec.until.watch_id) if spec.until else None
    if intent.stage == "verifying":
        if now - intent.released_at >= spec.verification_seconds:
            runtime._finish_intent("failed", "condition_not_observed")
            return
        if not _fresh(runtime, intent, [condition], now):
            return
        if condition.sample_started <= intent.released_at or condition.observation_id == intent.last_verification:
            return
        intent.last_verification = condition.observation_id
        # Stability is established by distinct fresh samples, not scheduler polls between them.
        if intent.condition_sample_started and condition.sample_started - intent.condition_sample_started > .1:
            intent.verified_since = None
        intent.condition_sample_started = condition.sample_started
        if condition.present != spec.until.present:
            intent.verified_since = None
            return
        if intent.verified_since is None:
            intent.verified_since = condition.sample_started
        if condition.sample_started - intent.verified_since >= spec.settle_ms / 1000:
            intent.condition_verified = True
            runtime._finish_intent("completed", "condition_observed_after_release", stop=False)
        return
    if intent.stage in ("pressing", "releasing") or runtime.pending:
        return
    source = runtime.watches.get(spec.watch_id)
    destination = runtime.watches.get(spec.destination_watch_id) if spec.destination_watch_id else None
    relevant = ([source] if intent.stage == "approaching" else
                [destination] if spec.destination_watch_id else [])
    if intent.stage == "approaching" and spec.destination_watch_id:
        relevant.append(destination)
    if spec.until:
        relevant.append(condition)
    # For a fixed destination, refresh the bound screen even when no watched target is needed.
    if not relevant:
        relevant = [source]
    if not _fresh(runtime, intent, relevant, now):
        return
    satisfied = bool(spec.until and condition.present == spec.until.present)
    if intent.stage == "approaching" and satisfied:
        runtime._finish_intent("failed", "condition_already_satisfied")
        return
    if intent.stage == "approaching" and destination is not None and not destination.present:
        runtime._finish_intent("failed", "target_lost")
        return
    if intent.stage == "dragging" and satisfied:
        _send(runtime, intent, "end_drag", now, relevant)
        return
    target = source if intent.stage == "approaching" else destination
    if target is not None and not target.present:
        runtime._finish_intent("failed", "target_lost")
        return
    goal = (target.detection["x"], target.detection["y"]) if target else spec.destination
    if now - intent.last_tick < .01:
        return
    dt = min(.02, now - intent.last_tick)
    intent.last_tick = now
    if _arrived(intent, goal, now):
        _send(runtime, intent, "begin_drag" if intent.stage == "approaching" else "end_drag",
              now, relevant, point=intent.cursor)
        return
    point = None
    if intent.cursor is not None and now - intent.cursor_time <= .05:
        point = runtime._pointer_step(intent, goal, dt, now, min(relevant, key=lambda w: w.sample_started))
    _send(runtime, intent, "pointer", now, relevant, point=point, goal=goal)


def _send(runtime, intent, operation, now, watches, *, point=None, goal=None):
    proof = min(w.sample_started for w in watches)
    evidence = min(watches, key=lambda w: w.sample_started)
    cid = uuid4().hex
    runtime.pending[cid] = {"intent_id": intent.id, "watch_id": intent.spec.watch_id,
        "observation_id": evidence.observation_id, "captured_mono": evidence.captured,
        "sample_started_mono": proof, "capture_source": evidence.source,
        "percept_ready_mono": now, "goal": list(goal) if goal else None}
    command = {"command_id": cid, "target": runtime.layout,
        "deadline": min(runtime.expires, intent.expires, proof + .05), "point": point}
    if intent.held or operation in ("begin_drag", "end_drag"):
        command.update(drag_id=intent.id, hold_until=min(runtime.expires, intent.expires, proof + .1))
    try:
        getattr(runtime.output, operation)(command)
        intent.commands += 1
        if operation == "begin_drag":
            intent.stage = "pressing"
        elif operation == "end_drag":
            intent.stage = "releasing"
    except Exception as exc:
        runtime.pending.pop(cid, None)
        runtime.halt(f"submit_failed: {exc}", degraded=True)
