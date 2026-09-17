"""Timer resolution and a fixed-rate ticker that records how late every tick was."""
from __future__ import annotations

import ctypes
import sys
import time


class TimerPeriod:
    """`with TimerPeriod(1):` asks Windows for 1 ms scheduler granularity (time.sleep is coarse otherwise)."""

    def __init__(self, ms: int = 1):
        self.ms = ms

    def __enter__(self):
        if sys.platform == 'win32':
            ctypes.windll.winmm.timeBeginPeriod(self.ms)
        return self

    def __exit__(self, *exc):
        if sys.platform == 'win32':
            ctypes.windll.winmm.timeEndPeriod(self.ms)


def percentiles(values, ps=(50, 90, 95, 99)) -> dict[str, float]:
    if not values:
        return {f'p{p}': float('nan') for p in ps} | {'max': float('nan'), 'n': 0}
    s = sorted(values)
    out = {}
    for p in ps:
        k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
        out[f'p{p}'] = s[k]
    out['max'] = s[-1]
    out['n'] = len(s)
    return out


class Ticker:
    """Fixed-rate loop. Sleeps until `spin_ms` before the deadline, then spins for precision.

    `wait()` returns the lateness of the tick in seconds and keeps every lateness for `stats()`.
    If the loop falls more than two periods behind it resynchronises instead of bursting to catch up:
    a reflex layer must never fire a queue of stale ticks.
    """

    def __init__(self, hz: float, spin_ms: float = 1.5):
        self.period = 1.0 / hz
        self.spin = spin_ms / 1000.0
        self.late: list[float] = []
        self.next: float | None = None
        self.resyncs = 0

    def wait(self) -> float:
        now = time.perf_counter()
        if self.next is None:
            self.next = now + self.period
        while True:
            now = time.perf_counter()
            remaining = self.next - now
            if remaining <= 0:
                break
            if remaining > self.spin:
                time.sleep(remaining - self.spin)
        late = now - self.next
        self.late.append(late)
        if late > 2 * self.period:
            self.resyncs += 1
            self.next = now + self.period
        else:
            self.next += self.period
        return late

    def stats(self) -> dict:
        ms = [v * 1000 for v in self.late]
        return percentiles(ms) | {'resyncs': self.resyncs, 'hz': 1.0 / self.period}
