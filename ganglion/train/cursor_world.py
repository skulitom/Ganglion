"""Seeded cursor plant with latency, gain, scale and target-motion variation.

Evaluator state stays in this headless simulator. The policy gets only the same
cursor/goal observations that the resident controller can obtain from pixels.
"""
import numpy as np


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
    def __init__(self, seeds, *, steps=160):
        self.seeds, self.steps = list(seeds), steps
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
        self.initial = self.cursor.copy()
        self.commands = []
        self.tick = 0

    def step(self, point):
        # Gain perturbs the requested correction once, at issue time. The queued
        # absolute coordinate is later applied directly, as with MoveAbsolute.
        # Scaling an old target's distance from the arrival-time cursor would
        # invent an unstable second feedback loop in the actuator.
        self.commands.append(self.cursor + self.gain[:, None]*(np.asarray(point)-self.cursor))
        indices = self.tick - self.delay
        desired = np.array([self.commands[t][i] if t >= 0 else self.initial[i] for i, t in enumerate(indices)])
        self.cursor = desired.copy()
        self.cursor = np.clip(self.cursor, self.rect[:, :2], self.rect[:, :2]+self.rect[:, 2:]-1)
        self.goal += self.velocity * .01
        low, high = np.full_like(self.goal, 35), self.rect[:, 2:] - 35
        bounced = (self.goal < low) | (self.goal > high)
        self.velocity[bounced] *= -1
        self.goal = np.clip(self.goal, low, high)
        self.tick += 1

    def teacher(self):
        return teacher_points(self.cursor, self.goal, self.speed, self.rect)

    def senses(self, previous=None):
        delta = (self.goal - self.cursor) / (self.speed[:, None]*.3)
        velocity = np.zeros_like(delta) if previous is None else (self.cursor-previous)/(.01*self.speed[:, None])
        zeros = lambda n: np.zeros((self.B, n), dtype=np.float32)
        motion = np.concatenate((np.tanh(velocity), zeros(1)), axis=1).astype(np.float32)
        return {"goal": np.concatenate((np.tanh(delta), zeros(1), np.tanh(np.linalg.norm(delta, axis=1))[:, None]), axis=1).astype(np.float32),
                "haltere": motion, "jo": motion, "lptc": zeros(6), "ocelli": zeros(3),
                "wing_cs": zeros(3), "compass": zeros(2)}
