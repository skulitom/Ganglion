"""Quiet-screen reach, slider drag, accepted drop, and rejected-drop fixtures."""
import numpy as np

from .target import World, TARGET_RGB

DESTINATION_RGB = (240, 190, 50)
CONDITION_RGB = (60, 130, 240)


class ManipulationWorld(World):
    def __init__(self):
        super().__init__()
        self.trial_id = None
        self.mode = "static_reach"
        self.source = [100, 160]
        self.held = False
        self.confirmed = False
        self.threshold = 330
        self.last_cursor = (0, 0)

    def reset(self, trial):
        self.trial_id, self.mode = trial["id"], trial["mode"]
        self.source = [100, 130 + trial["seed"] * 25]
        self.threshold = 310 + trial["seed"] * 20
        self.confirmed, self.held = False, False
        self.dirty = True
        self.emit("trial_started", trial_id=self.trial_id, mode=self.mode)

    def update(self):
        pass  # Deliberately no animation or redraw heartbeat.

    def rendered(self):
        if self.dirty:
            self.emit("render_submitted", trial_id=self.trial_id, source=self.source,
                      condition=self.confirmed, held=self.held)
            self.dirty = False

    def move(self, x, y):
        self.last_cursor = (x, y)
        if self.held:
            self.source = [x, y]
            if self.mode in ("drag_until", "reject"):
                self.confirmed = x >= self.threshold
            self.dirty = True

    def down(self, x, y):
        hit = abs(x - self.source[0]) <= 18 and abs(y - self.source[1]) <= 18
        self.emit("pointer_down", trial_id=self.trial_id, x=x, y=y, hit=hit)
        if not hit:
            self.emit("false_action", trial_id=self.trial_id)
            return
        if self.mode == "static_reach":
            self.confirmed = True
            self.emit("hit", trial_id=self.trial_id)
        else:
            self.held = True
        self.dirty = True

    def up(self, x, y):
        self.emit("pointer_up", trial_id=self.trial_id, x=x, y=y, was_held=self.held)
        if self.held:
            accepted = ((self.mode == "drag_until" and x >= self.threshold)
                        or (self.mode == "drop" and 470 <= x <= 570 and 140 <= y <= 220))
            self.held = False
            self.confirmed = accepted
            self.emit("drop_accepted" if accepted else "drop_rejected", trial_id=self.trial_id,
                      x=x, y=y, threshold=self.threshold, destination=[520, 180])
            if not accepted:
                self.source = [100, 155]
        self.dirty = True

    def render(self):
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = [30, 22, 18]
        frame[140:220, 470:570] = DESTINATION_RGB[::-1]
        frame[280:305, 40:600] = CONDITION_RGB[::-1] if self.confirmed else (65, 65, 110)
        x, y = (round(v) for v in self.source)
        if not (self.mode == "static_reach" and self.confirmed):
            frame[max(0, y - 18):y + 18, max(0, x - 18):x + 18] = TARGET_RGB[::-1]
        return frame


class SimulatedPointer:
    def __init__(self, capture):
        self.capture = capture
        self.cursor = (30, 100)
        self.held = False

    def cursor_pos(self):
        return self.cursor

    def move_abs(self, x, y):
        self.cursor = (x, y)
        with self.capture.lock:
            self.capture.world.move(x, y)

    def button(self, name, down):
        self.held = down
        with self.capture.lock:
            action = self.capture.world.down if down else self.capture.world.up
            action(*self.cursor)

    def release_all(self):
        if self.held:
            self.button("left", False)
