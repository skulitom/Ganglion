import ctypes
import sys
import time

import pytest

from ganglion.core import clock
from ganglion.core.actuators import INPUT, VK, abs_coords, utf16_units

win = pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')


def test_abs_coords_span_the_virtual_desktop():
    virtual = (-1920, 0, 3840, 1080)
    assert abs_coords(-1920, 0, virtual) == (0, 0)
    assert abs_coords(1919, 1079, virtual) == (65535, 65535)
    x, y = abs_coords(0, 540, virtual)
    assert abs(x - 32768) < 40 and abs(y - 32768) < 40
    assert abs_coords(99999, -5, virtual) == (65535, 0)  # clamped


def test_utf16_units_handle_astral_characters():
    assert utf16_units('ab') == [0x61, 0x62]
    assert utf16_units('\U0001F41D') == [0xD83D, 0xDC1D]  # a fly, as a surrogate pair


def test_key_table():
    assert VK['a'] == 0x41 and VK['f12'] == 0x7B and VK['escape'] == 0x1B and VK['9'] == 0x39


@win
def test_input_struct_size_matches_win32():
    assert ctypes.sizeof(INPUT) == 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28


def test_percentiles():
    p = clock.percentiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert p['p50'] == 5 or p['p50'] == 6
    assert p['max'] == 10 and p['n'] == 10
    assert clock.percentiles([])['n'] == 0


def test_ticker_holds_its_rate():
    with clock.TimerPeriod(1):
        tk = clock.Ticker(200)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.5:
            tk.wait()
    s = tk.stats()
    assert 80 <= s['n'] <= 120
    assert s['p99'] < 10.0  # ms late; loose, this is a correctness test not a benchmark


def test_ticker_resyncs_instead_of_bursting():
    tk = clock.Ticker(100)
    tk.wait()
    time.sleep(0.05)  # fall five periods behind
    tk.wait()
    assert tk.resyncs == 1
    late = tk.wait()
    assert late < 0.005


@win
def test_session_info():
    from ganglion.core import session
    info = session.current()
    assert info.session_id >= 0 and info.screen_w > 0 and info.screen_h > 0
    assert info.kind in ('console', 'child session (seat)') or info.kind.startswith('session ')


def test_capture_selects_primary_adapter_and_output(monkeypatch):
    from types import SimpleNamespace
    from ganglion.core.capture import primary_output
    monkeypatch.setitem(sys.modules, 'dxcam', SimpleNamespace(output_info=lambda: (
        'Device[0] Output[0]: Res:(1920, 1080) Rot:0 Primary:False\n'
        'Device[1] Output[2]: Res:(1280, 720) Rot:0 Primary:True\n')))
    assert primary_output() == (1, 2)
