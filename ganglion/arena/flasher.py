"""A window that changes colour on command or on input, and reports exactly when it did.

The Phase 0 stand-in for an application. `ganglion bench` flips it over UDP (display latency: flip
-> pixels in the capture), clicks it with SendInput (input latency: click -> event -> flip -> pixels)
and reads the time stamps it sends back. It runs at a fixed frame rate, like a game, so the loop
wait a real application adds is part of the measurement.

Protocol (UDP text on 127.0.0.1):
  bench -> flasher   `PING <t_ns>` | `FLIP [<id>]` | `QUIT`
  flasher -> bench   `READY <x> <y> <w> <h> <fps>`
                     `PONG <t_bench_ns> <t_flasher_ns>`
                     `FLIP <t_flip_ns> <cause> <t_event_ns> <n_pending> <new_state> <id>`
                     cause: cmd | click | key; id echoes the FLIP command (-1 for input)
Time stamps are `time.perf_counter_ns()` (QueryPerformanceCounter, one clock for every process on
the machine); the bench verifies that with PING/PONG before trusting cross-process differences.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import socket
import sys
import time

COMMAND_PORT = 9700
REPLY_PORT = 9701


def parse_reply(line: str) -> tuple[str, list]:
    """`FLIP 123 click 100 1` -> ('FLIP', [123, 'click', 100, 1]); numbers become ints."""
    parts = line.strip().split()
    if not parts:
        return '', []
    out = []
    for p in parts[1:]:
        try:
            out.append(int(p))
        except ValueError:
            out.append(p)
    return parts[0], out


def command_line(python: str, port: int, reply: int, x: int, y: int, w: int, h: int, fps: int,
                 seconds: float = 0.0) -> list[str]:
    return [python, '-m', 'ganglion.arena.flasher', '--port', str(port), '--reply', str(reply),
            '--x', str(x), '--y', str(y), '--w', str(w), '--h', str(h), '--fps', str(fps),
            '--seconds', str(seconds)]


def _client_rect(hwnd: int) -> tuple[int, int, int, int]:
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    rc = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rc))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return int(pt.x), int(pt.y), int(rc.right - rc.left), int(rc.bottom - rc.top)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=COMMAND_PORT)
    ap.add_argument('--reply', type=int, default=REPLY_PORT)
    ap.add_argument('--x', type=int, default=200)
    ap.add_argument('--y', type=int, default=200)
    ap.add_argument('--w', type=int, default=320)
    ap.add_argument('--h', type=int, default=240)
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--seconds', type=float, default=0.0, help='exit after this long (0 = until QUIT)')
    a = ap.parse_args(argv)

    from ..core.session import ensure_dpi_aware
    ensure_dpi_aware()
    os.environ['SDL_VIDEO_WINDOW_POS'] = f'{a.x},{a.y}'
    os.environ['SDL_MOUSE_FOCUS_CLICKTHROUGH'] = '1'   # the click that focuses us still counts
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')   # no audio: device enumeration can hang in a child session
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import pygame
    pygame.display.init()                               # video only, never the mixer
    print(f'flasher: pygame {pygame.version.ver}, driver {pygame.display.get_driver()}', flush=True)
    screen = pygame.display.set_mode((a.w, a.h), pygame.NOFRAME)
    print(f'flasher: window {a.w}x{a.h} at {a.x},{a.y}', flush=True)
    pygame.display.set_caption('ganglion-flasher')
    screen.fill((0, 0, 0))
    pygame.display.flip()
    try:
        hwnd = pygame.display.get_wm_info()['window']
        # stay on top: the bench clicks here and nothing else may receive those clicks. (A no-activate
        # style would keep the user's focus, but SDL then drops the button events; the bench restores
        # the previous foreground window instead.)
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)  # TOPMOST, NOMOVE|NOSIZE|SHOW
        rect = _client_rect(hwnd)
    except Exception:
        rect = (a.x, a.y, a.w, a.h)

    cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd.bind(('127.0.0.1', a.port))
    reply = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ('127.0.0.1', a.reply)

    def send(msg: str) -> None:
        try:
            reply.sendto(msg.encode(), dest)
        except OSError:
            pass

    # Commands are received on a thread so a PING is answered within microseconds (the bench
    # aligns clocks with it) and a FLIP carries its true receive time, not the frame loop's.
    import queue
    import threading
    inbox: queue.Queue = queue.Queue()

    def receiver() -> None:
        while True:
            try:
                data, _ = cmd.recvfrom(128)
            except OSError:
                return
            now = time.perf_counter_ns()
            parts = data.split()
            if data.startswith(b'PING'):
                send(f'PONG {parts[1].decode()} {now}')
            elif data.startswith(b'FLIP'):
                inbox.put(('cmd', now, int(parts[1]) if len(parts) > 1 else 0))
            elif data.startswith(b'QUIT'):
                inbox.put(('quit', now, 0))

    threading.Thread(target=receiver, name='flasher-receiver', daemon=True).start()

    send(f'READY {rect[0]} {rect[1]} {rect[2]} {rect[3]} {a.fps}')
    state = 0
    clock = pygame.time.Clock()
    t0 = time.perf_counter()
    running = True
    while running:
        pending: list[tuple[str, int, int]] = []
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                pending.append(('click', time.perf_counter_ns(), -1))
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                else:
                    pending.append(('key', time.perf_counter_ns(), -1))
        while True:
            try:
                cause, t_ns, cmd_id = inbox.get_nowait()
            except queue.Empty:
                break
            if cause == 'quit':
                running = False
            else:
                pending.append((cause, t_ns, cmd_id))
        if pending:
            state ^= 1
            screen.fill((255, 255, 255) if state else (0, 0, 0))
            pygame.display.flip()
            t_flip = time.perf_counter_ns()
            cause, t_ev, cmd_id = pending[-1]
            # FLIP <t_flip> <cause> <t_event> <n_pending> <new_state> <cmd_id>
            send(f'FLIP {t_flip} {cause} {t_ev} {len(pending)} {state} {cmd_id}')
        clock.tick(a.fps)
        if a.seconds and time.perf_counter() - t0 > a.seconds:
            running = False
    pygame.quit()
    return 0


if __name__ == '__main__':
    sys.exit(main())
