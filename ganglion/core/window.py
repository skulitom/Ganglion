"""Typed Win32 target identity and client-area checks; never changes focus."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

from .session import ensure_dpi_aware


def api():
    ensure_dpi_aware()
    u = ctypes.windll.user32
    u.GetForegroundWindow.restype = wintypes.HWND
    u.SetForegroundWindow.argtypes = [wintypes.HWND]
    u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    for name in ("IsWindow", "IsWindowVisible", "IsIconic"):
        getattr(u, name).argtypes = [wintypes.HWND]
    u.WindowFromPoint.argtypes = [wintypes.POINT]
    u.WindowFromPoint.restype = wintypes.HWND
    u.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    u.GetAncestor.restype = wintypes.HWND
    return u


def describe(hwnd: int) -> dict:
    u = api()
    if not u.IsWindow(hwnd) or not u.IsWindowVisible(hwnd) or u.IsIconic(hwnd):
        raise ValueError("target window is missing, hidden, or minimized")
    pid, rect, origin = wintypes.DWORD(), wintypes.RECT(), wintypes.POINT()
    u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not u.GetClientRect(hwnd, ctypes.byref(rect)) or not u.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise OSError("cannot read target client rectangle")
    name = ctypes.create_unicode_buffer(256)
    u.GetClassNameW(hwnd, name, 256)
    return {"hwnd": hwnd, "pid": pid.value, "class": name.value,
            "rect": [origin.x, origin.y, rect.right, rect.bottom]}


def find(*, pid: int | None = None, title: str | None = None) -> list[dict]:
    """Visible, unowned top-level windows of a process or with a title substring (never focuses)."""
    if pid is None and title is None:
        raise ValueError("select windows by pid or title")
    u = api()
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    u.EnumWindows.argtypes = [callback, wintypes.LPARAM]
    u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
    u.GetWindow.restype = wintypes.HWND
    matches = []

    def visit(hwnd, _):
        if not u.IsWindowVisible(hwnd) or u.GetWindow(hwnd, 4):  # GW_OWNER: skip owned popups
            return True
        owner = wintypes.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        text = ctypes.create_unicode_buffer(256)
        u.GetWindowTextW(hwnd, text, 256)
        if (pid is None or owner.value == pid) and (title is None or title.lower() in text.value.lower()):
            try:
                matches.append(describe(hwnd) | {"title": text.value})
            except (ValueError, OSError):
                pass
        return True

    u.EnumWindows(callback(visit), 0)
    return matches


def validate(target: dict, x=None, y=None, *, foreground=True):
    u = api()
    if describe(target["hwnd"]) != target:
        raise ValueError("target identity or layout changed; rebind after look")
    if foreground and u.GetForegroundWindow() != target["hwnd"]:
        raise ValueError("target is no longer the foreground window")
    if x is not None:
        tx, ty, w, h = target["rect"]
        if not tx <= x < tx + w or not ty <= y < ty + h:
            raise ValueError("click is outside the target")
        hit = u.WindowFromPoint(wintypes.POINT(x, y))
        if u.GetAncestor(hit, 2) != target["hwnd"]:  # GA_ROOT: allow child controls
            raise ValueError("target is occluded at the click location")
