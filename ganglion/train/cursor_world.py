"""Seeded cursor plant with latency, gain, scale and target-motion variation.

Evaluator state stays in this headless simulator. The policy gets only the same
cursor/goal observations that the resident controller can obtain from pixels.
"""
import numpy as np

from ..core.aim import UNBOUNDED

VIEW_LATENCY_TICKS = 2       # frames between a view delta and its effect in the capture
SLIP_WINDOW_TICKS = 6        # the flow percept compares frames 60 ms apart


def teacher_points(cursor, goal, speed, rect, dt=.01):
    low, high = rect[:, :2], rect[:, :2] + rect[:, 2:] - 1
    cursor = np.clip(cursor, low, high)
    delta = goal - cursor
    distance = np.linalg.norm(delta, axis=1)
    travel = np.minimum(np.minimum(distance, speed*min(.02, max(0, dt))),
                        np.maximum(1, distance*35*min(.02, max(0, dt))))
    fraction = np.divide(travel, distance, out=np.zeros_like(distance), where=distance > 0)
    return np.clip(np.rint(cursor + delta*fraction[:, None]), low, high)


class CursorWorld:
    def __init__(self, seeds, *, steps=160, jump_every=None, sense_version=2, view=False):
        if sense_version not in (2, 3, 4):
            raise ValueError("The training world speaks sensory adapter versions 2, 3 and 4")
        self.seeds, self.steps, self.sense_version = list(seeds), steps, sense_version
        self.B = len(self.seeds)
        rngs = [np.random.default_rng(seed) for seed in self.seeds]
        self.rect = np.array([[0, 0, r.integers(480, 1921), r.integers(320, 1081)] for r in rngs], dtype=float)
        self.speed = np.array([r.uniform(400, 1800) for r in rngs])
        self.gain = np.array([r.uniform(.6, 1.4) for r in rngs])
        self.delay = np.array([r.integers(0, 7) for r in rngs])
        self.cursor = np.array([r.uniform([30, 30], size-30) for r, size in zip(rngs, self.rect[:, 2:])])
        self.goal = np.array([r.uniform([40, 40], size-40) for r, size in zip(rngs, self.rect[:, 2:])])
        # Begin with bounded reaches, including near-target cases for settling.
        # Slower delayed plants receive reachable moving-target speeds.
        angles = np.array([r.uniform(-np.pi, np.pi) for r in rngs])
        radii = np.array([r.uniform(2, 60 if seed % 4 < 2 else 400) for seed, r in zip(self.seeds, rngs)])
        self.cursor = np.clip(self.goal + np.stack((np.cos(angles), np.sin(angles)), axis=1)*radii[:, None],
                              [0, 0], self.rect[:, 2:]-1)
        limits = np.minimum(150, self.speed*self.gain/(self.delay+1)*.2)
        self.velocity = np.array([r.uniform(-limit, limit, 2) if seed % 2 else [0, 0]
                                  for seed, r, limit in zip(self.seeds, rngs, limits)])
        if view is True or view is False:
            self.view = np.full(self.B, bool(view))
        else:
            self.view = np.asarray(view, dtype=bool)
        if self.view.any():
            # A view turned by relative deltas: unbounded, seen through the capture with extra
            # latency, and the target must start inside the frame or there is nothing to track.
            self.delay = self.delay + np.where(self.view, VIEW_LATENCY_TICKS, 0)
            cap = np.minimum(self.rect[:, 2], self.rect[:, 3]) / 2 - 60
            offset = self.cursor - self.goal
            norm = np.maximum(np.linalg.norm(offset, axis=1), 1e-9)
            scale = np.where(self.view, np.minimum(1, cap / norm), 1)
            self.cursor = self.goal + offset * scale[:, None]
        self.initial = self.cursor.copy()
        self.commands = []
        self.history = []            # cursor after each step, for the visual slip
        self.tick = 0
        # Absolute commands from a stale cursor advance at speed*gain/(delay+1); a jump the
        # reference covers in about 1.5 s keeps re-acquisition comparable across plants.
        self.reach_cap = np.minimum(400, self.speed * self.gain / (self.delay + 1) * 1.5)
        self.jump_every, self.jumps = jump_every, 0
        self.jump_rngs = [np.random.default_rng(seed + 2_000_003) for seed in self.seeds]

    def jump(self):
        """Move every target 40 px to its reach cap away, inside the area: a fresh reach from
        whatever state the controller has reached, which a settled episode never shows it."""
        self.jumps += 1
        angles = np.array([r.uniform(-np.pi, np.pi) for r in self.jump_rngs])
        radii = np.array([r.uniform(40, cap) for r, cap in zip(self.jump_rngs, self.reach_cap)])
        goal = self.goal + np.stack((np.cos(angles), np.sin(angles)), axis=1) * radii[:, None]
        self.goal = np.clip(goal, 40, self.rect[:, 2:] - 40)

    @property
    def bounds(self):
        """Where commands and the envelope are clamped: the client area, or nothing for a view."""
        bounds = self.rect.copy()
        bounds[self.view] = UNBOUNDED
        return bounds

    def step(self, point):
        if self.jump_every and self.tick and self.tick % self.jump_every == 0:
            self.jump()
        # Gain perturbs the requested correction once, at issue time. The queued
        # absolute coordinate is later applied directly, as with MoveAbsolute.
        # Scaling an old target's distance from the arrival-time cursor would
        # invent an unstable second feedback loop in the actuator.
        self.commands.append(self.cursor + self.gain[:, None]*(np.asarray(point)-self.cursor))
        indices = self.tick - self.delay
        desired = np.array([self.commands[t][i] if t >= 0 else self.initial[i] for i, t in enumerate(indices)])
        clipped = np.clip(desired, self.rect[:, :2], self.rect[:, :2]+self.rect[:, 2:]-1)
        self.cursor = np.where(self.view[:, None], desired, clipped)
        self.history.append(self.cursor.copy())
        del self.history[:-SLIP_WINDOW_TICKS - VIEW_LATENCY_TICKS - 2]
        self.goal += self.velocity * .01
        low, high = np.full_like(self.goal, 35), self.rect[:, 2:] - 35
        bounced = (self.goal < low) | (self.goal > high)
        self.velocity[bounced] *= -1
        self.goal = np.clip(self.goal, low, high)
        self.tick += 1

    def teacher(self):
        return teacher_points(self.cursor, self.goal, self.speed, self.bounds)

    def slip(self):
        """What the wide-field flow percept would report for a view episode: the image translation
        in px/s over the last 60 ms, seen two frames late, which is minus the view's own motion.
        Zero for cursor episodes, where the picture does not move."""
        slip = np.zeros_like(self.cursor)
        needed = SLIP_WINDOW_TICKS + VIEW_LATENCY_TICKS + 1
        if len(self.history) >= needed:
            recent = self.history[-VIEW_LATENCY_TICKS - 1]
            older = self.history[-VIEW_LATENCY_TICKS - 1 - SLIP_WINDOW_TICKS]
            slip = -(recent - older) / (SLIP_WINDOW_TICKS * .01)
        return np.where(self.view[:, None], slip, 0)

    def senses(self, previous=None):
        """The runtime adapter's channels for every episode: goal error scaled by 0.3 s of intent
        speed; version 2 adds the cursor's own velocity in units of that speed; version 4 feeds
        the visual slip of view episodes to the lptc channel instead."""
        delta = (self.goal - self.cursor) / (self.speed[:, None]*.3)
        still = previous is None or self.sense_version in (3, 4)
        velocity = np.zeros_like(delta) if still else (self.cursor-previous)/(.01*self.speed[:, None])
        zeros = lambda n: np.zeros((self.B, n), dtype=np.float32)
        motion = np.concatenate((np.tanh(velocity), zeros(1)), axis=1).astype(np.float32)
        lptc = zeros(6)
        if self.sense_version == 4 and self.view.any():
            lptc[:, :2] = np.tanh(self.slip() / self.speed[:, None])
        return {"goal": np.concatenate((np.tanh(delta), zeros(1), np.tanh(np.linalg.norm(delta, axis=1))[:, None]), axis=1).astype(np.float32),
                "haltere": motion, "jo": motion, "lptc": lptc, "ocelli": zeros(3),
                "wing_cs": zeros(3), "compass": zeros(2)}
