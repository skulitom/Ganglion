import numpy as np
import pytest

from ganglion.core.capture import Capture
from ganglion.core import gdi


def test_static_refresh_acquires_new_pixels_and_labels_them(monkeypatch):
    capture = Capture(refresh_static=True, refresh_interval=0, poll_sleep=0)
    capture.width = capture.height = 2
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    fresh = np.full_like(first, 77)

    class Camera:
        count = 0
        def grab(self):
            self.count += 1
            if self.count == 1:
                return first
            return None

    class Refresh:
        closed = False
        def __init__(self, width, height):
            assert (width, height) == (2, 2)
        def grab(self):
            capture._stop.set()
            return fresh
        def close(self):
            Refresh.closed = True

    capture._cam = Camera()
    monkeypatch.setattr(gdi, "GDIRefresh", Refresh)
    capture._loop()
    frame, available, seq, source, started = capture.wait_observation(0)
    assert np.array_equal(frame, fresh) and seq == 2 and source == "gdi_refresh"
    assert started <= available and capture.source_counts == {"dxgi": 1, "gdi_refresh": 1}
    assert Refresh.closed and capture.error is None


def test_refresh_failure_never_advances_a_cached_frame(monkeypatch):
    capture = Capture(refresh_static=True, refresh_interval=0, poll_sleep=0)
    class Camera:
        def grab(self):
            return None
    class Refresh:
        def __init__(self, *args):
            pass
        def grab(self):
            raise OSError("display unavailable")
        def close(self):
            pass
    capture._cam = Camera()
    capture._frame, capture._t, capture._seq = np.zeros((2, 2, 3)), 10, 7
    monkeypatch.setattr(gdi, "GDIRefresh", Refresh)
    capture._loop()
    assert capture.latest()[1:] == (10, 7)
    assert "display unavailable" in capture.error


def test_raw_benchmark_capture_does_not_enable_fallback(monkeypatch):
    capture = Capture(poll_sleep=0)
    class Camera:
        def grab(self):
            capture._stop.set()
            return None
    def forbidden(*args):
        pytest.fail("benchmark capture must remain DXGI only")
    capture._cam = Camera()
    monkeypatch.setattr(gdi, "GDIRefresh", forbidden)
    capture._loop()
    assert capture.latest() == (None, 0, 0)


def test_slow_acquisition_does_not_become_fresh_by_finishing_late():
    from test_reach import setup
    c, output, runtime, layout, wid = setup()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[100:130, 100:130] = [120, 220, 40]
    runtime.observe(frame, c(), 2, layout, source="gdi_refresh", sample_started=c() - .2)
    assert not output.commands
    assert not runtime.watches[wid].present
    assert runtime.snapshot()["capture_source"] == "gdi_refresh"
