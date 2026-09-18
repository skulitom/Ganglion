"""Which Windows session this process runs in, and the screen it sees.

The fast loop must run inside the session where the application lives (console, or an the seat tool child
session). Everything here is a thin ctypes layer; it never changes anything.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

_dpi_done = False


def ensure_dpi_aware() -> None:
    """Ask for physical pixels from GetSystemMetrics/GetCursorPos (per-monitor v2, else system DPI)."""
    global _dpi_done
    if _dpi_done or sys.platform != 'win32':
        return
    _dpi_done = True
    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


@dataclass(frozen=True)
class SessionInfo:
    session_id: int
    console_session_id: int
    is_child_session: bool
    screen_w: int
    screen_h: int
    virtual: tuple[int, int, int, int]  # x, y, w, h of the virtual desktop

    @property
    def kind(self) -> str:
        if self.is_child_session:
            return 'child session (seat)'
        if self.session_id == self.console_session_id:
            return 'console'
        return f'session {self.session_id} (not the console)'


def current() -> SessionInfo:
    ensure_dpi_aware()
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    sid = wintypes.DWORD(0)
    kernel32.ProcessIdToSessionId(kernel32.GetCurrentProcessId(), ctypes.byref(sid))
    console = int(kernel32.WTSGetActiveConsoleSessionId())
    child = wintypes.BOOL(False)
    try:
        ctypes.windll.wtsapi32.WTSIsChildSession(ctypes.byref(child))
    except Exception:
        pass
    sm = user32.GetSystemMetrics
    return SessionInfo(
        session_id=int(sid.value), console_session_id=console, is_child_session=bool(child.value),
        screen_w=int(sm(0)), screen_h=int(sm(1)),
        virtual=(int(sm(76)), int(sm(77)), int(sm(78)), int(sm(79))),
    )


def foreground_window_class() -> str:
    """Class name of the foreground window (a `GameInputServiceWindow` here blocks SendInput)."""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ''
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value
