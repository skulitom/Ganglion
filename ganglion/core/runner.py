"""Resident capture/perception/tick service. The transport never owns the control loop."""
from __future__ import annotations

import json
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from .clock import TimerPeriod
from .protocol import Server, Service
from .runtime import Runtime


class Runner:
    def __init__(self, runtime, capture, target):
        self.runtime, self.capture, self.target = runtime, capture, target
        self.stop_event = threading.Event()
        self.threads = []

    def start(self):
        # Pay native import/worker initialization before exposing control to the agent.
        import cv2
        import numpy as np
        cv2.setNumThreads(1)
        cv2.connectedComponentsWithStats(np.zeros((8, 8), dtype=np.uint8), connectivity=8)
        for name, function in (("vision", self._vision), ("tick", self._tick)):
            thread = threading.Thread(target=function, name=f"ganglion-{name}", daemon=True)
            thread.start()
            self.threads.append(thread)
        return self

    def _vision(self):
        seq = 0
        try:
            while not self.stop_event.is_set():
                if hasattr(self.capture, "wait_observation"):
                    frame, captured, observation, source, started = self.capture.wait_observation(seq, timeout=0.1)
                else:
                    frame, captured, observation = self.capture.wait_new(seq, timeout=0.1)
                    source = "provided"
                    started = captured
                if frame is not None and observation > seq:
                    seq = observation
                    self.runtime.observe(frame, captured, seq, self.target(), source=source, sample_started=started)
        except Exception as exc:
            self.runtime.halt(f"vision_failed: {exc}", degraded=True)
            self.stop_event.set()

    def _tick(self):
        try:
            with TimerPeriod(1):
                deadline = time.perf_counter()
                while not self.stop_event.is_set():
                    self.runtime.tick()
                    if self.capture.error or time.perf_counter() - self.capture.last_poll > 0.5:
                        self.runtime.halt(self.capture.error or "capture_stalled", degraded=True)
                        self.stop_event.set()
                        break
                    with self.runtime.lock:
                        active = self.runtime.owner is not None
                        bound = self.runtime.layout
                    if active and bound is not None:
                        current = self.target()
                        if current != bound:
                            self.runtime.halt("target_layout_changed", degraded=True)
                        elif bound.get("hwnd"):
                            from .window import validate
                            validate(bound)
                    deadline = max(deadline + 0.01, time.perf_counter())
                    self.stop_event.wait(max(0, deadline - time.perf_counter()))
        except Exception as exc:
            self.runtime.halt(f"tick_failed: {exc}", degraded=True)
            self.stop_event.set()

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(2)
        self.runtime.halt("core_stopped")
        if self.runtime.shadow:
            self.runtime.shadow.close()


def run_core(endpoint_path, *, hwnd=None, synthetic=False, seconds=0, shadow_checkpoint=None):
    from .output import MemoryOutput, ProcessOutput
    endpoint_path = Path(endpoint_path)
    if endpoint_path.exists():
        raise ValueError("Endpoint file exists; choose a new path or remove a confirmed stale endpoint.")
    with ExitStack() as stack:
        predictor = None
        if shadow_checkpoint:
            from ganglion.brain.haltere_cursor import HaltereCursor
            predictor = HaltereCursor(shadow_checkpoint)
        if synthetic:
            from ganglion.arena.target import SyntheticArena
            capture = SyntheticArena()
            output = MemoryOutput(time.perf_counter, capture.click)
            target, session_id = capture.target, 0
        else:
            from .capture import Capture
            from .session import current
            from .window import describe
            if hwnd is None:
                raise ValueError("Live core requires --window HWND; use --synthetic for a headless run.")
            info = current()
            capture, output = Capture(refresh_static=True), ProcessOutput()
            target, session_id = lambda: describe(hwnd), info.session_id
        stack.callback(output.close)
        capture.start()
        stack.callback(capture.stop)
        runtime = Runtime(time.perf_counter, output, session_id=session_id, shadow_predictor=predictor)
        runner = Runner(runtime, capture, target).start()
        stack.callback(runner.close)
        service = Service(runtime)
        server = Server(service)
        stack.callback(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        stack.callback(server.shutdown)
        stack.callback(service.closed.set)
        endpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with endpoint_path.open("x", encoding="utf-8") as stream:
            json.dump(server.endpoint(), stream)
        stack.callback(endpoint_path.unlink, missing_ok=True)
        print(f"Ganglion core ready; endpoint: {endpoint_path}", flush=True)
        deadline = time.perf_counter() + seconds if seconds else float("inf")
        try:
            while time.perf_counter() < deadline and not runner.stop_event.wait(0.1):
                pass
        except KeyboardInterrupt:
            pass
        failed = runtime.state == "degraded"
    return 1 if failed else 0
