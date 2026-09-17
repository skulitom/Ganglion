"""Input injection with SendInput: mouse (absolute over the virtual desktop, or relative deltas for
games that read raw input), buttons, wheel, keys as scan codes, unicode text. Tracks what is held so
`release_all()` can always return the machine to a neutral state (the dead-man's job).

No sleeps live here: timing belongs to the scheduler that calls these primitives.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from .session import ensure_dpi_aware

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOUSEEVENTF_MOVE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_VIRTUALDESK = 0x0001, 0x8000, 0x4000
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP = 0x0020, 0x0040
MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP = 0x0080, 0x0100
MOUSEEVENTF_WHEEL, MOUSEEVENTF_HWHEEL = 0x0800, 0x1000
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, KEYEVENTF_SCANCODE = 0x0001, 0x0002, 0x0004, 0x0008
WHEEL_DELTA = 120

ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG), ('mouseData', wintypes.DWORD),
                ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD), ('dwExtraInfo', ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [('wVk', wintypes.WORD), ('wScan', wintypes.WORD), ('dwFlags', wintypes.DWORD),
                ('time', wintypes.DWORD), ('dwExtraInfo', ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [('uMsg', wintypes.DWORD), ('wParamL', wintypes.WORD), ('wParamH', wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ('u',)
    _fields_ = [('type', wintypes.DWORD), ('u', _INPUTUNION)]


BUTTON_FLAGS = {
    'left': (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, 0),
    'right': (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, 0),
    'middle': (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, 0),
    'x1': (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, 1),
    'x2': (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, 2),
}

# Virtual-key codes by name; letters, digits and F-keys are added below.
VK = {
    'backspace': 0x08, 'tab': 0x09, 'enter': 0x0D, 'return': 0x0D, 'shift': 0x10, 'ctrl': 0x11, 'alt': 0x12,
    'pause': 0x13, 'capslock': 0x14, 'escape': 0x1B, 'esc': 0x1B, 'space': 0x20, 'pageup': 0x21,
    'pagedown': 0x22, 'end': 0x23, 'home': 0x24, 'left': 0x25, 'up': 0x26, 'right': 0x27, 'down': 0x28,
    'printscreen': 0x2C, 'insert': 0x2D, 'delete': 0x2E, 'win': 0x5B, 'lwin': 0x5B, 'rwin': 0x5C, 'apps': 0x5D,
    'numpad0': 0x60, 'numpad1': 0x61, 'numpad2': 0x62, 'numpad3': 0x63, 'numpad4': 0x64, 'numpad5': 0x65,
    'numpad6': 0x66, 'numpad7': 0x67, 'numpad8': 0x68, 'numpad9': 0x69, 'multiply': 0x6A, 'add': 0x6B,
    'subtract': 0x6D, 'decimal': 0x6E, 'divide': 0x6F, 'numlock': 0x90, 'scrolllock': 0x91,
    'lshift': 0xA0, 'rshift': 0xA1, 'lctrl': 0xA2, 'rctrl': 0xA3, 'lalt': 0xA4, 'ralt': 0xA5,
    'semicolon': 0xBA, 'equals': 0xBB, 'comma': 0xBC, 'minus': 0xBD, 'period': 0xBE, 'slash': 0xBF,
    'grave': 0xC0, 'tilde': 0xC0, 'lbracket': 0xDB, 'backslash': 0xDC, 'rbracket': 0xDD, 'quote': 0xDE,
}
for _i in range(1, 25):
    VK[f'f{_i}'] = 0x6F + _i
for _c in 'abcdefghijklmnopqrstuvwxyz':
    VK[_c] = ord(_c.upper())
for _d in '0123456789':
    VK[_d] = ord(_d)

# Keys whose scan code needs the extended flag (navigation cluster, right-hand modifiers, Win, Apps).
EXTENDED = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2C, 0x2D, 0x2E, 0x5B, 0x5C, 0x5D, 0x90,
            0xA3, 0xA5, 0x6F}


def abs_coords(x: int, y: int, virtual: tuple[int, int, int, int]) -> tuple[int, int]:
    """Screen pixel -> 0..65535 normalised coordinates over the virtual desktop (pure, for tests)."""
    vx, vy, vw, vh = virtual
    nx = round((x - vx) * 65535 / max(1, vw - 1))
    ny = round((y - vy) * 65535 / max(1, vh - 1))
    return min(65535, max(0, nx)), min(65535, max(0, ny))


def utf16_units(s: str) -> list[int]:
    """UTF-16 code units of a string, the currency of KEYEVENTF_UNICODE (pure, for tests)."""
    b = s.encode('utf-16-le')
    return [int.from_bytes(b[i:i + 2], 'little') for i in range(0, len(b), 2)]


class Actuators:
    def __init__(self):
        ensure_dpi_aware()
        self.user32 = ctypes.windll.user32 if sys.platform == 'win32' else None
        self.held_buttons: set[str] = set()
        self.held_keys: set[int] = set()
        self.sent = 0

    # -- low level -------------------------------------------------------------------------------
    def _send(self, *inputs: INPUT) -> int:
        if self.user32 is None:
            raise OSError("SendInput requires Windows")
        arr = (INPUT * len(inputs))(*inputs)
        n = self.user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
        self.sent += n
        if n != len(inputs):
            raise OSError(f"SendInput inserted {n}/{len(inputs)} events (blocked or partial submission)")
        return n

    @staticmethod
    def _mouse(dx=0, dy=0, data=0, flags=0) -> INPUT:
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(dx, dy, data, flags, 0, 0)
        return inp

    @staticmethod
    def _key(scan=0, vk=0, flags=0) -> INPUT:
        inp = INPUT(type=INPUT_KEYBOARD)
        inp.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
        return inp

    # -- mouse -----------------------------------------------------------------------------------
    def virtual_desktop(self) -> tuple[int, int, int, int]:
        sm = self.user32.GetSystemMetrics
        return int(sm(76)), int(sm(77)), int(sm(78)), int(sm(79))

    def cursor_pos(self) -> tuple[int, int]:
        pt = wintypes.POINT()
        if not self.user32.GetCursorPos(ctypes.byref(pt)):
            raise OSError("GetCursorPos failed; cursor position is unavailable")
        return int(pt.x), int(pt.y)

    def move_abs(self, x: int, y: int) -> None:
        nx, ny = abs_coords(x, y, self.virtual_desktop())
        self._send(self._mouse(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK))

    def move_rel(self, dx: int, dy: int) -> None:
        self._send(self._mouse(int(dx), int(dy), 0, MOUSEEVENTF_MOVE))

    def button(self, name: str, down: bool) -> None:
        d, u, data = BUTTON_FLAGS[name]
        self._send(self._mouse(0, 0, data, d if down else u))
        (self.held_buttons.add if down else self.held_buttons.discard)(name)

    def wheel(self, notches: float, horizontal: bool = False) -> None:
        self._send(self._mouse(0, 0, int(round(notches * WHEEL_DELTA)) & 0xFFFFFFFF,
                               MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL))

    # -- keyboard --------------------------------------------------------------------------------
    def key(self, name_or_vk: str | int, down: bool) -> None:
        vk = VK[name_or_vk.lower()] if isinstance(name_or_vk, str) else int(name_or_vk)
        scan = self.user32.MapVirtualKeyW(vk, 0)  # MAPVK_VK_TO_VSC
        flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED else 0) | (0 if down else KEYEVENTF_KEYUP)
        self._send(self._key(scan, 0, flags))
        (self.held_keys.add if down else self.held_keys.discard)(vk)

    def text(self, s: str) -> None:
        inputs = []
        for code in utf16_units(s):
            inputs.append(self._key(code, 0, KEYEVENTF_UNICODE))
            inputs.append(self._key(code, 0, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        if inputs:
            self._send(*inputs)

    # -- safety ----------------------------------------------------------------------------------
    def release_all(self) -> None:
        failures = []
        for b in list(self.held_buttons):
            try:
                self.button(b, False)
            except Exception as exc:
                failures.append(str(exc))
        for vk in list(self.held_keys):
            try:
                self.key(vk, False)
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            raise OSError("; ".join(failures))
