"""Evaluator-only moving targets. The policy receives pixels, never this state."""
import math

from .target import World


class ReachWorld(World):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trial_id = None
        self.seed = 0
        self.started = self.clock()
        self.done = True

    def reset(self, trial):
        self.trial_id, self.seed = trial["id"], trial["seed"]
        self.target_id += 1
        self.started = self.clock()
        self.done = False
        self.emit("trial_started", trial_id=self.trial_id, seed=self.seed)
        self.update()

    def update(self):
        if self.done:
            self.box = None
            return
        t = self.clock() - self.started
        phase = self.seed * 1.17
        x = round(320 + 170 * math.sin(t * 0.9 + phase))
        y = round(180 + 75 * math.cos(t * 1.1 + phase))
        self.box = (x - 18, y - 18, 36, 36)
        self.dirty = True

    def click(self, x, y):
        hit = self.box is not None and (self.box[0] <= x < self.box[0] + self.box[2]
                                       and self.box[1] <= y < self.box[1] + self.box[3])
        self.emit("input_received", trial_id=self.trial_id, x=x, y=y, box=self.box)
        self.emit("hit" if hit else "false_action", trial_id=self.trial_id, x=x, y=y)
        if hit:
            self.done = True
            self.box = None
            self.dirty = True
