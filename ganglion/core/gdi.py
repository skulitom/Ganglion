"""Fresh primary-display samples for quiet DXGI periods; no cached-pixel timestamps."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

import numpy as np


class BitmapInfo(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("width", wintypes.LONG), ("height", wintypes.LONG),
                ("planes", wintypes.WORD), ("bits", wintypes.WORD), ("compression", wintypes.DWORD),
                ("image_size", wintypes.DWORD), ("xppm", wintypes.LONG), ("yppm", wintypes.LONG),
                ("used", wintypes.DWORD), ("important", wintypes.DWORD)]


class GDIRefresh:
    """Reusable top-down DIB. Create, sample and close on the capture thread."""
    def __init__(self, width, height):
        from .session import ensure_dpi_aware
        ensure_dpi_aware()
        self.width, self.height = width, height
        self.user, self.gdi = ctypes.windll.user32, ctypes.windll.gdi32
        u, g = self.user, self.gdi
        u.GetDC.argtypes, u.GetDC.restype = [wintypes.HWND], wintypes.HDC
        u.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        g.CreateCompatibleDC.argtypes, g.CreateCompatibleDC.restype = [wintypes.HDC], wintypes.HDC
        g.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.POINTER(BitmapInfo), wintypes.UINT,
                                      ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
        g.CreateDIBSection.restype = wintypes.HBITMAP
        g.SelectObject.argtypes, g.SelectObject.restype = [wintypes.HDC, wintypes.HANDLE], wintypes.HANDLE
        g.DeleteObject.argtypes, g.DeleteDC.argtypes = [wintypes.HANDLE], [wintypes.HDC]
        g.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
        self.screen = self.memory = self.bitmap = self.previous = None
        try:
            self.screen = u.GetDC(None)
            self.memory = g.CreateCompatibleDC(self.screen)
            if not self.screen or not self.memory:
                raise OSError("GDI refresh could not create a display DC")
            info = BitmapInfo(ctypes.sizeof(BitmapInfo), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
            bits = ctypes.c_void_p()
            self.bitmap = g.CreateDIBSection(self.screen, ctypes.byref(info), 0, ctypes.byref(bits), None, 0)
            if not self.bitmap or not bits.value:
                raise OSError("GDI refresh could not allocate pixels")
            self.previous = g.SelectObject(self.memory, self.bitmap)
            if not self.previous or self.previous == ctypes.c_void_p(-1).value:
                self.previous = None
                raise OSError("GDI refresh could not select its bitmap")
            self.pixels = np.ctypeslib.as_array((ctypes.c_ubyte * (width * height * 4)).from_address(bits.value))
        except BaseException:
            self.close()
            raise

    def grab(self):
        if (self.user.GetSystemMetrics(0), self.user.GetSystemMetrics(1)) != (self.width, self.height):
            raise OSError("primary-display dimensions changed; restart capture")
        if not self.gdi.BitBlt(self.memory, 0, 0, self.width, self.height, self.screen, 0, 0, 0x40CC0020):
            raise OSError("GDI static refresh BitBlt failed")
        # GDI drawing can be batched; flush before reading DIB memory.
        if not self.gdi.GdiFlush():
            raise OSError("GDI static refresh flush failed")
        return self.pixels.reshape(self.height, self.width, 4)[:, :, :3].copy()

    def close(self):
        if self.previous and self.memory:
            self.gdi.SelectObject(self.memory, self.previous)
        if self.bitmap:
            self.gdi.DeleteObject(self.bitmap)
        if self.memory:
            self.gdi.DeleteDC(self.memory)
        if self.screen:
            self.user.ReleaseDC(None, self.screen)
        self.screen = self.memory = self.bitmap = self.previous = None
