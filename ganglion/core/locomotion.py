"""Continuous key locomotion as a bounded program that the runtime renews at frame rate.

A move holds up to four keys until a taught condition holds, its duration ends, it is
cancelled, or the lease/target fails. The helper never holds a key longer than one renewal
window, so a stalled runtime lets the keys go within a fraction of a second; the runtime renews
them every 100 ms while the program is alive. Keys are an actuator of their own, so a move can
run alongside a pointer program such as align.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

RENEW_EVERY = 0.1
HOLD_WINDOW = 0.35


@dataclass
class Move:
    id: str
    spec: object
    started: float
    expires: float
    phase: str = "running"
    reason: str | None = None
    last_renew: float = float("-inf")
    renewals: int = 0

    @property
    def active(self):
        return self.phase == "running"

    def status(self):
        return {"intent_id": self.id, "program": "move", "keys": list(self.spec.keys), "phase": self.phase,
                "reason": self.reason, "started_mono": self.started, "expires_mono": self.expires,
                "renewals": self.renewals, "until": self.spec.until.model_dump() if self.spec.until else None,
                "task_success_verified": False}


def drive_move(runtime, move, now):
    spec = move.spec
    if now >= move.expires:
        finish_move(runtime, move, "completed", "duration")
        return
    if spec.until is not None:
        watch = runtime.watches.get(spec.until.watch_id)
        if watch is None:
            finish_move(runtime, move, "failed", "watch_removed")
            return
        if watch.observation_id and now - watch.sample_started <= 0.25 and watch.present == spec.until.present:
            finish_move(runtime, move, "completed", "condition_observed")
            return
    if now - move.last_renew < RENEW_EVERY:
        return
    move.last_renew = now
    cid = uuid4().hex
    hold_until = min(runtime.expires, move.expires, now + HOLD_WINDOW)
    runtime.key_pending[cid] = {"intent_id": move.id, "action": "move"}
    try:
        runtime.output.key({"command_id": cid, "target": runtime.layout, "keys": list(spec.keys),
                            "deadline": min(runtime.expires, now + 0.05), "hold_until": hold_until})
        move.renewals += 1
        runtime.note_self_motion(hold_until + 0.35)
    except Exception as exc:
        runtime.key_pending.pop(cid, None)
        runtime.halt(f"submit_failed: {exc}", degraded=True)


def finish_move(runtime, move, phase, reason):
    if not move.active:
        return
    move.phase, move.reason = phase, reason
    runtime.ledger.append("intent_" + phase, runtime.clock(), critical=True, **move.status())
    cid = uuid4().hex
    runtime.key_pending[cid] = {"intent_id": move.id, "action": "release"}
    try:
        runtime.output.release({"command_id": cid, "keys": list(move.spec.keys)})
    except Exception as exc:
        runtime.key_pending.pop(cid, None)
        runtime.halt(f"release_failed: {exc}", degraded=True)
