"""`ganglion doctor`: what is installed, which session we are in, and what would block the fast loop.

Read-only. Every check is tolerant: a missing optional piece is a WARN with a fix, not a crash.
"""
from __future__ import annotations

import ctypes
import importlib
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Check:
    name: str
    state: str          # OK | WARN | FAIL | SKIP
    detail: str = ''
    fix: str = ''


def _reg(root, path: str, name: str):
    import winreg
    try:
        with winreg.OpenKey(root, path) as k:
            v, _ = winreg.QueryValueEx(k, name)
            return v
    except OSError:
        return None


def run(quick: bool = False) -> list[Check]:
    checks: list[Check] = []
    checks.append(Check('python', 'OK', f'{sys.version.split()[0]} at {sys.executable}'))

    # session and screen
    try:
        from .core import session as sess
        info = sess.current()
        checks.append(Check('session', 'OK', f"{info.kind}; id {info.session_id}, console {info.console_session_id}; "
                                             f"screen {info.screen_w}x{info.screen_h}, virtual {info.virtual}"))
        cls = sess.foreground_window_class()
        if cls == 'GameInputServiceWindow':
            checks.append(Check('foreground', 'FAIL', 'GameInputServiceWindow is in front: SendInput is blocked',
                                'see Anode scripts/repair-seat-input.ps1'))
        else:
            checks.append(Check('foreground', 'OK', f'class {cls!r}'))
    except Exception as e:
        checks.append(Check('session', 'FAIL', f'{type(e).__name__}: {e}'))

    # seat frame cap and GPU policy (machine registry, read only)
    import winreg
    fi = _reg(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Terminal Server\WinStations', 'DWMFRAMEINTERVAL')
    if fi is None:
        checks.append(Check('seat frame cap', 'WARN', 'DWMFRAMEINTERVAL unset: child sessions capped at 30 fps',
                            'run scripts/set-seat-fps.ps1 as administrator, then reboot'))
    else:
        checks.append(Check('seat frame cap', 'OK' if fi <= 10 else 'WARN', f'DWMFRAMEINTERVAL = {fi} ms (~{1000 / fi:.0f} fps cap)',
                            '' if fi <= 10 else 'scripts/set-seat-fps.ps1 sets 10 (~100 fps); reboot to apply'))
    hw = _reg(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services', 'bEnumerateHWBeforeSW')
    checks.append(Check('seat GPU policy', 'OK' if hw == 1 else 'WARN', f'bEnumerateHWBeforeSW = {hw}',
                        '' if hw == 1 else 'anode setup --gpu (child sessions render on the software adapter otherwise)'))

    # pointer settings that change how relative mouse deltas land
    try:
        user32 = ctypes.windll.user32
        mouse = (ctypes.c_int * 3)()
        user32.SystemParametersInfoW(0x0003, 0, mouse, 0)  # SPI_GETMOUSE
        speed = ctypes.c_int(0)
        user32.SystemParametersInfoW(0x0070, 0, ctypes.byref(speed), 0)  # SPI_GETMOUSESPEED
        epp = 'on' if mouse[2] else 'off'
        checks.append(Check('pointer', 'OK' if not mouse[2] else 'WARN',
                            f'enhance pointer precision {epp}, speed {speed.value}/20',
                            '' if not mouse[2] else 'acceleration makes relative moves nonlinear; the Arena models it, games with raw input ignore it'))
    except Exception as e:
        checks.append(Check('pointer', 'WARN', f'{type(e).__name__}: {e}'))

    # timer granularity
    try:
        from .core.clock import TimerPeriod
        with TimerPeriod(1):
            t0 = time.perf_counter()
            for _ in range(20):
                time.sleep(0.001)
            per = (time.perf_counter() - t0) / 20 * 1000
        checks.append(Check('timer', 'OK' if per < 2.5 else 'WARN', f'sleep(1 ms) takes {per:.2f} ms with timeBeginPeriod(1)'))
    except Exception as e:
        checks.append(Check('timer', 'WARN', f'{type(e).__name__}: {e}'))

    # capture
    try:
        import dxcam
        from .core.capture import primary_output
        device_idx, idx = primary_output()
        detail = dxcam.output_info().strip().replace('\n', '; ')
        if quick:
            checks.append(Check('capture', 'OK', f'dxcam outputs: {detail}; primary idx {idx}'))
        else:
            cam = dxcam.create(device_idx=device_idx, output_idx=idx, output_color='BGR')
            f = cam.grab()
            t0 = time.perf_counter()
            for _ in range(50):
                cam.grab()
            cost = (time.perf_counter() - t0) / 50 * 1000
            cam.release()
            shape = None if f is None else f.shape
            checks.append(Check('capture', 'OK', f'DXGI duplication on output {idx}: first frame {shape}, poll {cost:.3f} ms; outputs: {detail}'))
    except Exception as e:
        checks.append(Check('capture', 'FAIL', f'{type(e).__name__}: {e}', 'pip install dxcam; exclusive-fullscreen or a suspended RDP viewer can also break duplication'))

    # torch / CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else 'no CUDA'
        checks.append(Check('torch', 'OK' if cuda else 'WARN', f'{torch.__version__}, {dev}',
                            '' if cuda else 'the 30k-neuron brain needs the GPU for 100 Hz (72 ms/step on CPU)'))
    except Exception as e:
        checks.append(Check('torch', 'WARN', f'{type(e).__name__}: {e}', 'uv pip install torch --index-url https://download.pytorch.org/whl/cu128'))

    # Haltere brain
    try:
        import haltere  # noqa: F401
        from .bench.run import CHECKPOINT, HALTERE_ROOT
        graph = HALTERE_ROOT / 'data' / 'built' / 'flight.npz'
        st = 'OK' if CHECKPOINT.exists() and graph.exists() else 'WARN'
        checks.append(Check('haltere', st, f'package at {Path(haltere.__file__).parent}; checkpoint {CHECKPOINT.exists()}, graph {graph.exists()}',
                            '' if st == 'OK' else 'haltere build / a slim checkpoint in Haltere/artifacts'))
    except Exception as e:
        checks.append(Check('haltere', 'WARN', f'{type(e).__name__}: {e}', r'uv pip install -e C:\DEV\Haltere'))

    # gamepad
    svc = _reg(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Services\ViGEmBus', 'ImagePath')
    try:
        importlib.import_module('vgamepad')
        checks.append(Check('gamepad', 'OK' if svc else 'WARN', f'vgamepad importable; ViGEmBus service {"present" if svc else "missing"}',
                            '' if svc else 'install ViGEmBus 1.22.0'))
    except Exception as e:
        checks.append(Check('gamepad', 'WARN', f'vgamepad: {type(e).__name__}: {e}; ViGEmBus service {"present" if svc else "missing"}',
                            'set VGAMEPAD_SKIP_VIGEMBUS_INSTALL=true; uv pip install vgamepad'))

    for mod, fix in (('mcp', 'uv pip install mcp'), ('pygame', 'uv pip install pygame'), ('cv2', 'uv pip install opencv-python')):
        try:
            m = importlib.import_module(mod)
            checks.append(Check(mod, 'OK', getattr(m, '__version__', '') or getattr(getattr(m, 'version', None), 'ver', '')))
        except Exception as e:
            checks.append(Check(mod, 'WARN', f'{type(e).__name__}: {e}', fix))
    return checks


def format_checks(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        line = f'{c.state:4} {c.name.ljust(width)}  {c.detail}'
        if c.fix:
            line += f'\n     {" " * width}  fix: {c.fix}'
        lines.append(line)
    worst = 'FAIL' if any(c.state == 'FAIL' for c in checks) else ('WARN' if any(c.state == 'WARN' for c in checks) else 'OK')
    lines.append(f'=> {worst}')
    return '\n'.join(lines)


def to_json(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]
