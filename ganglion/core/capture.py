"""Screen capture through DXGI desktop duplication (dxcam), newest-frame semantics.

A thread polls the duplication; `latest()` returns the most recent frame with its time stamp and
sequence number, never a queue. Works on the console and inside an Anode child session (measured
2026-09-16: 66 fresh frames/s in the seat, 0.06 ms per poll).
"""
from __future__ import annotations

import re
import threading
import time

import numpy as np


def primary_output_idx() -> int:
    """dxcam output index of the primary display (its pixel coordinates match mouse injection)."""
    return primary_output()[1]


def primary_output() -> tuple[int, int]:
    """Return both adapter and output indices; output indices are per-adapter."""
    import dxcam
    info = dxcam.output_info()
    for m in re.finditer(r'Device\[(\d+)\] Output\[(\d+)\]:.*?Primary:True', info):
        return int(m.group(1)), int(m.group(2))
    raise RuntimeError("dxcam did not identify the primary display")


class Capture:
    def __init__(self, output_idx: int | None = None, color: str = 'BGR', poll_sleep: float = 0.0005,
                 copy: bool = True, refresh_static: bool = False, refresh_interval: float = 0.02):
        self.output_idx = output_idx
        self.color = color
        self.poll_sleep = poll_sleep
        self.copy = copy
        if refresh_static and (color != 'BGR' or output_idx is not None):
            raise ValueError("static refresh requires BGR on the automatically selected primary display")
        self.refresh_static, self.refresh_interval = refresh_static, refresh_interval
        self._source = "dxgi"
        self._sample_started = 0.0
        self.source_counts = {"dxgi": 0, "gdi_refresh": 0}
        self._cam = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._seq = 0
        self.polls = 0
        self.intervals: list[float] = []
        self.width = self.height = 0
        self.error: str | None = None
        self.last_poll = 0.0

    # -- lifecycle -------------------------------------------------------------------------------
    def start(self) -> 'Capture':
        import dxcam
        device_idx = 0
        if self.output_idx is None:
            device_idx, self.output_idx = primary_output()
        self._cam = dxcam.create(device_idx=device_idx, output_idx=self.output_idx, output_color=self.color)
        self.width, self.height = self._cam.width, self._cam.height
        self._stop.clear()
        self.error = None
        self.last_poll = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name='ganglion-capture', daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("capture thread is still using the camera")
        if self._cam is not None:
            try:
                self._cam.release()
            except Exception:
                pass
            self._cam = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- the poll loop ---------------------------------------------------------------------------
    def _loop(self) -> None:
        last_t = None
        refresh = None
        try:
            while not self._stop.is_set():
                sample_started = time.perf_counter()
                frame = self._cam.grab()
                self.last_poll = time.perf_counter()
                self.polls += 1
                source = "dxgi"
                if frame is None and self.refresh_static and self.last_poll - self._sample_started >= self.refresh_interval:
                    if refresh is None:
                        from .gdi import GDIRefresh
                        refresh = GDIRefresh(self.width, self.height)
                    # This is a new acquisition, not a relabelled cached DXGI frame.
                    sample_started = time.perf_counter()
                    frame, source = refresh.grab(), "gdi_refresh"
                if frame is None:
                    if self.poll_sleep:
                        time.sleep(self.poll_sleep)
                    continue
                if self.copy and source == "dxgi":
                    frame = frame.copy()
                t = time.perf_counter()
                with self._lock:
                    self._frame, self._t, self._seq, self._source = frame, t, self._seq + 1, source
                    self._sample_started = sample_started
                    self.source_counts[source] += 1
                # Keep the original DXGI interval measurement separate from fallback sampling.
                if source == "dxgi":
                    if last_t is not None:
                        self.intervals.append(t - last_t)
                        if len(self.intervals) > 100000:
                            del self.intervals[:50000]
                    last_t = t
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if refresh:
                refresh.close()

    # -- consumers -------------------------------------------------------------------------------
    def latest(self) -> tuple[np.ndarray | None, float, int]:
        with self._lock:
            return self._frame, self._t, self._seq

    def wait_new(self, seq: int, timeout: float = 1.0) -> tuple[np.ndarray | None, float, int]:
        return self.wait_observation(seq, timeout)[:3]

    def wait_observation(self, seq, timeout=1.0):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                f, t, s, source, started = self._frame, self._t, self._seq, self._source, self._sample_started
            if s > seq:
                return f, t, s, source, started
            time.sleep(0.0002)
        return None, 0.0, seq, None, 0.0

    @staticmethod
    def region_mean(frame: np.ndarray, rect: tuple[int, int, int, int], step: int = 4) -> float:
        x, y, w, h = rect
        sub = frame[y:y + h:step, x:x + w:step]
        return float(sub.mean()) if sub.size else float('nan')
