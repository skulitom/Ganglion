"""Deterministic cursor feedback controller. No application semantics or desktop APIs."""
from dataclasses import dataclass
from math import hypot

from .schema import IntentSpec


def step(cursor, target, dt, speed, rect):
    """Proportional correction, speed limited inside the bound client area.

    A cursor outside the area enters at its nearest edge. Scheduler stalls never
    accumulate a large movement: at most 20 ms of motion is integrated per step.
    """
    x, y, w, h = rect
    cx = min(x + w - 1, max(x, cursor[0]))
    cy = min(y + h - 1, max(y, cursor[1]))
    dx, dy = target[0] - cx, target[1] - cy
    distance = hypot(dx, dy)
    if distance == 0:
        return round(cx), round(cy)
    dt = min(0.02, max(0, dt))
    travel = min(distance, speed * dt, max(1, distance * 35 * dt))
    return (min(x + w - 1, max(x, round(cx + dx * travel / distance))),
            min(y + h - 1, max(y, round(cy + dy * travel / distance))))


@dataclass
class Reach:
    id: str
    spec: IntentSpec
    started: float
    expires: float
    phase: str = "running"
    reason: str | None = None
    last_tick: float = float("-inf")
    cursor: tuple | None = None
    cursor_time: float = 0.0
    consumed_cursor_time: float = 0.0
    settled_since: float | None = None
    error_px: float | None = None
    commands: int = 0

    @property
    def active(self):
        return self.phase in ("running", "clicking")

    def status(self):
        return {"intent_id": self.id, "program": self.spec.program, "watch_id": self.spec.watch_id,
                "phase": self.phase, "reason": self.reason, "started_mono": self.started,
                "expires_mono": self.expires, "cursor": self.cursor, "error_px": self.error_px,
                "commands": self.commands, "task_success_verified": False}
