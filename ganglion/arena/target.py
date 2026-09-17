"""Gate B target world with synthetic capture and a real pygame window.

Ground truth is evaluator-only. The runtime sees BGR pixels and ordinary input.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

TARGET_RGB = (40, 220, 120)


def activate_own_window(hwnd, *, fixture_pid=None):
    """Focus the benchmark's own fixture, including a separately launched browser process."""
    import ctypes
    from ctypes import wintypes
    from ganglion.core.window import api, describe
    if describe(hwnd)["pid"] != (fixture_pid or os.getpid()):
        raise ValueError("Arena activation only accepts its own window")
    u = api()

    def focused():
        # A cross-thread activation can be processed after SetForegroundWindow returns.
        until = time.perf_counter() + .25
        while time.perf_counter() < until:
            if u.GetForegroundWindow() == hwnd:
                return True
            time.sleep(.01)
        return False

    u.SetForegroundWindow(hwnd)
    if focused():
        return
    foreground = u.GetForegroundWindow()
    owner = wintypes.DWORD()
    other = u.GetWindowThreadProcessId(foreground, ctypes.byref(owner))
    here = ctypes.windll.kernel32.GetCurrentThreadId()
    u.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    target_thread = u.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
    attached = []
    try:
        for thread in {other, target_thread} - {0, here}:
            if u.AttachThreadInput(here, thread, True):
                attached.append(thread)
        u.SetForegroundWindow(hwnd)
    finally:
        for thread in reversed(attached):
            u.AttachThreadInput(here, thread, False)
    if not focused():
        raise RuntimeError("Arena could not acquire focus; no input test was started")


class World:
    def __init__(self, width=640, height=360, seed=17, clock=time.perf_counter):
        self.width, self.height, self.clock = width, height, clock
        self.rng = random.Random(seed)
        self.events = deque(maxlen=10000)
        self.target_id = 0
        self.box = None
        self.next_spawn = clock() + 0.6
        self.visible_until = 0.0
        self.dirty = True

    def emit(self, kind, **data):
        self.events.append({"kind": kind, "t_mono": self.clock(), "target_id": self.target_id, **data})

    def update(self):
        now = self.clock()
        if self.box is not None and now >= self.visible_until:
            self.emit("miss")
            self.box = None
            self.dirty = True
            self.next_spawn = now + self.rng.uniform(0.25, 0.55)
        if self.box is None and now >= self.next_spawn:
            self.target_id += 1
            x, y = self.rng.randrange(40, self.width - 96), self.rng.randrange(40, self.height - 96)
            self.box = (x, y, 56, 56)
            self.visible_until = now + 1.0
            self.emit("event_scheduled", box=self.box)
            self.dirty = True

    def click(self, x, y):
        self.emit("input_received", x=x, y=y)
        hit = self.box is not None and (self.box[0] <= x < self.box[0] + self.box[2]
                                       and self.box[1] <= y < self.box[1] + self.box[3])
        self.emit("hit" if hit else "false_action", x=x, y=y)
        if hit:
            self.box = None
            self.dirty = True
            self.next_spawn = self.clock() + self.rng.uniform(0.25, 0.55)

    def render(self):
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (30, 22, 18)  # BGR
        if self.box is not None:
            x, y, w, h = self.box
            frame[y:y + h, x:x + w] = TARGET_RGB[::-1]
        return frame

    def rendered(self):
        if self.dirty:
            self.emit("render_submitted", visible=self.box is not None)
            self.dirty = False


class SyntheticArena:
    """60 Hz replayable capture source; no Windows calls or GPU."""
    def __init__(self, seed=17, world=None):
        self.world = world or World(seed=seed)
        self.lock = threading.RLock()
        self.seq, self.captured, self.frame = 0, 0.0, None
        self.error, self.last_poll = None, 0.0
        self.stopped = threading.Event()
        self.thread = None

    def target(self):
        return {"hwnd": 0, "pid": 0, "class": "synthetic",
                "rect": [0, 0, self.world.width, self.world.height]}

    def start(self):
        self.last_poll = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stopped.is_set():
            with self.lock:
                self.world.update()
                self.frame = self.world.render()
                self.world.rendered()
                self.captured = self.last_poll = time.perf_counter()
                self.seq += 1
            self.stopped.wait(1 / 60)

    def click(self, x, y):
        with self.lock:
            self.world.click(x, y)

    def wait_new(self, seq, timeout=0.1):
        deadline = time.perf_counter() + timeout
        while not self.stopped.is_set() and time.perf_counter() < deadline:
            with self.lock:
                if self.seq > seq:
                    return self.frame, self.captured, self.seq
            self.stopped.wait(0.001)
        return None, 0, seq

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ready", required=True, help="write window metadata for the evaluator")
    parser.add_argument("--truth", required=True, help="write evaluator-only JSON ground truth on exit")
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--trial-file", help="evaluator-only moving-target trial reset file")
    parser.add_argument("--scenario", choices=["reach", "manipulation"], default="reach")
    args = parser.parse_args(argv)
    from ganglion.core.session import ensure_dpi_aware
    from ganglion.core.window import describe
    ensure_dpi_aware()
    os.environ["SDL_VIDEO_WINDOW_POS"] = "200,180"
    os.environ["SDL_MOUSE_FOCUS_CLICKTHROUGH"] = "1"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    pygame.display.init()
    if args.scenario == "manipulation":
        from .manipulation import ManipulationWorld
        world = ManipulationWorld()
    elif args.trial_file:
        from .reach_world import ReachWorld
        world = ReachWorld()
    else:
        world = World(seed=args.seed)
    screen = pygame.display.set_mode((world.width, world.height), pygame.NOFRAME)
    pygame.display.set_caption("Ganglion Arena — target reflex")
    hwnd = pygame.display.get_wm_info()["window"]
    # Only our own test window; the demo records/restores prior foreground and cursor.
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, wintypes.UINT]
    u.SetForegroundWindow.argtypes = [wintypes.HWND]
    u.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
    u.SetForegroundWindow(hwnd)
    activate_own_window(hwnd)
    frame_clock = pygame.time.Clock()
    deadline = time.perf_counter() + args.seconds
    ready = False
    try:
        running = True
        while running and time.perf_counter() < deadline and not Path(args.stop_file).exists():
            if args.trial_file and Path(args.trial_file).exists():
                trial = json.loads(Path(args.trial_file).read_text())
                if trial["id"] != world.trial_id:
                    world.reset(trial)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if args.scenario == "manipulation":
                        world.down(*event.pos)
                    else:
                        world.click(*event.pos)
                elif args.scenario == "manipulation" and event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    world.up(*event.pos)
                elif args.scenario == "manipulation" and event.type == pygame.MOUSEMOTION:
                    world.move(*event.pos)
            world.update()
            if world.dirty or args.scenario != "manipulation":
                bgr = world.render()
                pygame.surfarray.blit_array(screen, bgr[:, :, ::-1].transpose(1, 0, 2))
                pygame.display.flip()
                world.rendered()
            if not ready:
                Path(args.ready).write_text(json.dumps(describe(hwnd)))
                ready = True
            frame_clock.tick(60)
    finally:
        Path(args.truth).write_text(json.dumps(list(world.events), indent=2))
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
