"""Deterministic cursor feedback controller. No application semantics or desktop APIs."""
from dataclasses import dataclass
from math import hypot, isfinite

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


def supervise(cursor, goal, velocity, dt, speed, rect, tolerance=6.0):
    """Bounded acceptance of a proposed velocity in units of the intent speed.

    The step is clamped to the deterministic speed limit and the client area and must bring the
    cursor closer to the goal (or hold position once within tolerance). Anything else returns
    None and the deterministic controller acts for that tick. Pure, for tests.
    """
    try:
        vx, vy = float(velocity[0]), float(velocity[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (isfinite(vx) and isfinite(vy)):
        return None
    dt = min(0.02, max(0, dt))
    limit = speed * dt
    sx, sy = vx * limit, vy * limit
    length = hypot(sx, sy)
    if length > limit:
        sx, sy = sx * limit / length, sy * limit / length
    x, y, w, h = rect
    cx = min(x + w - 1, max(x, cursor[0]))
    cy = min(y + h - 1, max(y, cursor[1]))
    nx = min(x + w - 1, max(x, round(cx + sx)))
    ny = min(y + h - 1, max(y, round(cy + sy)))
    before = hypot(goal[0] - cx, goal[1] - cy)
    after = hypot(goal[0] - nx, goal[1] - ny)
    if before > tolerance and after >= before:
        return None
    if after > before + 0.5:
        return None
    return nx, ny


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
    neural_commands: int = 0
    overridden_commands: int = 0
    last_controller: str = "deterministic"

    @property
    def active(self):
        return self.phase in ("running", "clicking")

    def status(self):
        return {"intent_id": self.id, "program": self.spec.program, "watch_id": self.spec.watch_id,
                "phase": self.phase, "reason": self.reason, "started_mono": self.started,
                "expires_mono": self.expires, "cursor": self.cursor, "error_px": self.error_px,
                "commands": self.commands, "task_success_verified": False,
                "controller": self.spec.controller,
                "actuation_authority": "supervised_connectome" if self.spec.controller == "connectome" else "deterministic",
                "neural_commands": self.neural_commands, "overridden_commands": self.overridden_commands}
