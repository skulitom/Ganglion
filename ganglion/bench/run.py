"""The Phase 0 bench: capture rate and latency, input-to-pixel latency, brain step time, tick jitter.

Run it on the console and inside an Anode seat; the two JSON files are the Phase 0 deliverable.
Everything is measured against the flasher (ganglion.arena.flasher), a fixed-frame-rate window that
reports when it flipped, so the numbers include what a real application adds.
"""
from __future__ import annotations

import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..core import session as sess
from ..core.actuators import Actuators
from ..core.capture import Capture
from ..core.clock import TimerPeriod, Ticker, percentiles
from ..arena import flasher as fl

HALTERE_ROOT = Path(os.environ.get('GANGLION_HALTERE', r'C:\DEV\Haltere'))
CHECKPOINT = HALTERE_ROOT / 'artifacts' / 'ftPath2_best.pt'


class Flasher:
    """Runs the flasher as a subprocess and talks its UDP protocol."""

    def __init__(self, fps: int, x: int = 120, y: int = 120, w: int = 320, h: int = 240,
                 port: int = fl.COMMAND_PORT, reply: int = fl.REPLY_PORT, log_dir: Path | None = None):
        self.fps, self.port = fps, port
        self.reply = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.reply.bind(('127.0.0.1', reply))
        self.reply.settimeout(10.0)
        self.cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        log_dir = Path(log_dir or os.environ.get('TEMP', '.'))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f'flasher_{fps}.log'
        self._log = open(self.log_path, 'w', encoding='utf-8')
        self.proc = subprocess.Popen(fl.command_line(sys.executable, port, reply, x, y, w, h, fps),
                                     stdout=self._log, stderr=subprocess.STDOUT)
        try:
            kind, args = self.recv('READY')
        except (socket.timeout, TimeoutError):
            self.proc.kill()
            self._log.close()
            tail = self.log_path.read_text(encoding='utf-8', errors='replace')[-2000:]
            raise RuntimeError(f'flasher window did not report READY within 10 s (exit {self.proc.poll()}); '
                               f'log {self.log_path}:\n{tail}') from None
        self.rect = tuple(int(v) for v in args[:4])
        self.state = 0
        self.clock_offset_ns = None

    def send(self, msg: str) -> None:
        self.cmd.sendto(msg.encode(), ('127.0.0.1', self.port))

    def recv(self, expect: str | None = None, timeout: float = 5.0) -> tuple[str, list]:
        self.reply.settimeout(timeout)
        while True:
            data, _ = self.reply.recvfrom(256)
            kind, args = fl.parse_reply(data.decode(errors='replace'))
            if expect is None or kind == expect:
                return kind, args

    def drain(self) -> None:
        self.reply.settimeout(0.0)
        try:
            while True:
                self.reply.recvfrom(256)
        except (BlockingIOError, OSError):
            pass

    def settle(self, seconds: float = 0.2) -> None:
        """Let queued flips finish and discard their replies, so a trial never reads a stale one."""
        time.sleep(seconds)
        self.drain()

    def flip_reply(self, cmd_id: int, timeout: float = 2.0) -> list | None:
        """Wait for the FLIP reply that echoes `cmd_id` (a stale reply is skipped); None on timeout."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                _, args = self.recv('FLIP', timeout=max(0.01, deadline - time.perf_counter()))
            except (socket.timeout, TimeoutError):
                return None
            if len(args) >= 6 and args[5] == cmd_id:
                return args
        return None

    def input_reply(self, t_after_ns: int, timeout: float = 1.0) -> list | None:
        """Wait for a FLIP reply caused by input whose event time is after `t_after_ns` (our clock)."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                _, args = self.recv('FLIP', timeout=max(0.01, deadline - time.perf_counter()))
            except (socket.timeout, TimeoutError):
                return None
            if len(args) >= 6 and args[1] in ('click', 'key') and args[2] - self.clock_offset_ns >= t_after_ns - 1_000_000:
                return args
        return None

    def check_clock(self, n: int = 40) -> dict:
        """PING/PONG: the flasher's clock minus ours, from the round trips with the least delay.

        perf_counter is QueryPerformanceCounter on Windows, one clock for every process, so the
        offset should be within the round-trip time (tens of microseconds); anything larger means the
        time stamps are not comparable and the bench falls back to its own clock only.
        """
        samples = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            self.send(f'PING {t0}')
            _, args = self.recv('PONG')
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0, args[1] - (t0 + t1) / 2))
        samples.sort()
        best = samples[:max(3, n // 4)]
        offset_ns = statistics.median(o for _, o in best)
        rtt_us = best[0][0] / 1e3
        self.clock_offset_ns = offset_ns if abs(offset_ns) < 2e6 else 0.0
        return {'offset_ms': offset_ns / 1e6, 'min_rtt_us': rtt_us, 'comparable': abs(offset_ns) < 2e6}

    def close(self) -> None:
        try:
            self.send('QUIT')
            self.proc.wait(timeout=3.0)
        except Exception:
            self.proc.kill()
        self.reply.close()
        self.cmd.close()
        try:
            self._log.close()
        except Exception:
            pass


def _provenance(**args) -> dict:
    """What produced these numbers: versions, machine, command (kept next to every result)."""
    import platform
    from importlib.metadata import version, PackageNotFoundError
    from .. import __version__

    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None
    out = {'ganglion': __version__, 'python': sys.version.split()[0], 'machine': platform.node(),
           'windows': platform.version(), 'args': args, 'argv': sys.argv,
           'packages': {p: ver(p) for p in ('torch', 'dxcam', 'pygame', 'numpy', 'haltere', 'mcp')}}
    try:
        import torch
        out['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out['gpu'] = None
    return out


def _inner(rect):
    x, y, w, h = rect
    return (x + w // 4, y + h // 4, w // 2, h // 2)


def _detect(cap: Capture, region, want_white: bool, seq_from: int, timeout: float = 1.0):
    """Wait for a frame after `seq_from` whose region mean matches the wanted colour; return (t, seq)."""
    deadline = time.perf_counter() + timeout
    seq = seq_from
    while time.perf_counter() < deadline:
        frame, t, s = cap.wait_new(seq, timeout=max(0.0, deadline - time.perf_counter()))
        if frame is None:
            break
        seq = s
        m = cap.region_mean(frame, region)
        if (m > 128) == want_white:
            return t, s
    return None, seq


def bench_capture_rate(cap: Capture, flasher: Flasher, seconds: float = 3.0) -> dict:
    """Flip the flasher as fast as it will go and count fresh captured frames."""
    flasher.drain()
    _, _, seq0 = cap.latest()
    polls0 = cap.polls
    t0 = time.perf_counter()
    n_cmd = 0
    while time.perf_counter() - t0 < seconds:
        flasher.send('FLIP 0')
        n_cmd += 1
        time.sleep(1.0 / (flasher.fps * 2))
    dt = time.perf_counter() - t0
    _, _, seq1 = cap.latest()
    flasher.settle()
    intervals = [v * 1000 for v in cap.intervals[-(seq1 - seq0):]] if seq1 > seq0 else []
    return {'flasher_fps': flasher.fps, 'fresh_frames_per_s': (seq1 - seq0) / dt,
            'polls_per_s': (cap.polls - polls0) / dt, 'interval_ms': percentiles(intervals)}


def bench_display_latency(cap: Capture, flasher: Flasher, trials: int) -> dict:
    """FLIP command -> flasher flips -> the change shows up in the capture."""
    region = _inner(flasher.rect)
    flasher.settle()
    cmd_to_flip, flip_to_pixel, cmd_to_pixel = [], [], []
    misses = 0
    for i in range(trials):
        _, _, seq = cap.latest()
        t_send = time.perf_counter()
        flasher.send(f'FLIP {1000 + i}')
        args = flasher.flip_reply(1000 + i)
        if args is None:
            misses += 1
            continue
        t_flip = (args[0] - flasher.clock_offset_ns) / 1e9
        t_det, seq = _detect(cap, region, bool(args[4]), seq)
        if t_det is None:
            misses += 1
            continue
        cmd_to_flip.append((t_flip - t_send) * 1000)
        flip_to_pixel.append((t_det - t_flip) * 1000)
        cmd_to_pixel.append((t_det - t_send) * 1000)
        time.sleep(0.05)
    return {'trials': trials, 'misses': misses, 'cmd_to_flip_ms': percentiles(cmd_to_flip),
            'flip_to_pixel_ms': percentiles(flip_to_pixel), 'cmd_to_pixel_ms': percentiles(cmd_to_pixel)}


def bench_input_latency(cap: Capture, flasher: Flasher, act: Actuators, trials: int) -> dict:
    """SendInput click -> the flasher's event loop sees it -> it flips -> the change is captured."""
    import ctypes
    region = _inner(flasher.rect)
    x, y, w, h = flasher.rect
    user32 = ctypes.windll.user32
    hwnd_before = user32.GetForegroundWindow()
    act.move_abs(x + w // 2, y + h // 2)
    time.sleep(0.2)
    flasher.settle()
    click_to_event, event_to_flip, flip_to_pixel, click_to_pixel = [], [], [], []
    misses = 0
    for _ in range(trials):
        _, _, seq = cap.latest()
        t_click = time.perf_counter()
        act.button('left', True)
        act.button('left', False)
        args = flasher.input_reply(int(t_click * 1e9))
        if args is None:
            misses += 1
            continue
        t_flip = (args[0] - flasher.clock_offset_ns) / 1e9
        t_ev = (args[2] - flasher.clock_offset_ns) / 1e9
        t_det, seq = _detect(cap, region, bool(args[4]), seq)
        if t_det is None:
            misses += 1
            continue
        click_to_event.append((t_ev - t_click) * 1000)
        event_to_flip.append((t_flip - t_ev) * 1000)
        flip_to_pixel.append((t_det - t_flip) * 1000)
        click_to_pixel.append((t_det - t_click) * 1000)
        time.sleep(0.05)
    try:  # give the user their window back (we sent the last input, so we are allowed to)
        if hwnd_before:
            user32.SetForegroundWindow(hwnd_before)
    except Exception:
        pass
    return {'trials': trials, 'misses': misses, 'click_to_event_ms': percentiles(click_to_event),
            'event_to_flip_ms': percentiles(event_to_flip), 'flip_to_pixel_ms': percentiles(flip_to_pixel),
            'click_to_pixel_ms': percentiles(click_to_pixel)}


def bench_cursor(act: Actuators, n: int = 200) -> dict:
    """SendInput absolute move -> GetCursorPos read-back: exactness and cost."""
    vx, vy, vw, vh = act.virtual_desktop()
    rng = np.random.default_rng(0)
    errs, costs = [], []
    start = act.cursor_pos()
    for _ in range(n):
        x = int(rng.integers(vx + 10, vx + vw - 10))
        y = int(rng.integers(vy + 10, vy + vh - 10))
        t0 = time.perf_counter()
        act.move_abs(x, y)
        px, py = act.cursor_pos()
        costs.append((time.perf_counter() - t0) * 1e6)
        errs.append(max(abs(px - x), abs(py - y)))
    act.move_abs(*start)
    return {'n': n, 'max_error_px': int(max(errs)), 'mean_error_px': float(np.mean(errs)),
            'move_and_read_us': percentiles(costs)}


def bench_tick(hz: float, seconds: float, work=None) -> dict:
    with TimerPeriod(1):
        tk = Ticker(hz)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            tk.wait()
            if work is not None:
                work()
    return tk.stats()


def load_brain(device: str = 'cuda'):
    """Haltere's shipped brain; None if Haltere or its checkpoint is not available."""
    if not CHECKPOINT.exists():
        return None
    try:
        from haltere.train.bptt import load_checkpoint
    except Exception:
        return None
    brain, cfg, graph = load_checkpoint(str(CHECKPOINT), device)
    brain.eval()
    return brain


def bench_brain(device: str = 'cuda', batches=(1, 2, 4), steps: int = 200) -> dict:
    import torch
    t_load = time.perf_counter()
    brain = load_brain(device)
    if brain is None:
        return {'available': False}
    out = {'available': True, 'checkpoint': str(CHECKPOINT), 'device': str(brain.device),
           'load_s': time.perf_counter() - t_load, 'neurons': int(brain.N),
           'edges': int(brain.edge_index.shape[1]), 'step_ms': {}}
    W = brain.weight_matrix().detach()
    with torch.no_grad():
        for B in batches:
            state = brain.init_state(B)
            obs = {k: torch.zeros(B, d, device=brain.device) for k, d in brain.channel_dims.items()}
            for _ in range(20):
                act, state, aux = brain(obs, state, W)
            if brain.device.type == 'cuda':
                torch.cuda.synchronize()
            times = []
            for _ in range(steps):
                t0 = time.perf_counter()
                act, state, aux = brain(obs, state, W)
                if brain.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            out['step_ms'][str(B)] = percentiles(times)
    return out


def run(trials: int = 30, fps: int = 60, brain: bool = True, cpu: bool = False, label: str | None = None,
        tick_seconds: float = 3.0) -> dict:
    info = sess.current()
    results = {
        'label': label or info.kind, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'session': {'id': info.session_id, 'console': info.console_session_id, 'child': info.is_child_session,
                    'kind': info.kind, 'screen': [info.screen_w, info.screen_h], 'virtual': list(info.virtual)},
        'provenance': _provenance(trials=trials, fps=fps, brain=brain, cpu=cpu, tick_seconds=tick_seconds),
    }
    act = Actuators()

    # capture: poll cost first (own duplication, released before the capture thread starts)
    import dxcam
    from ..core.capture import primary_output_idx
    idx = primary_output_idx()
    cam = dxcam.create(output_idx=idx, output_color='BGR')
    cam.grab()
    costs = []
    for _ in range(200):
        t0 = time.perf_counter()
        cam.grab()
        costs.append((time.perf_counter() - t0) * 1000)
    cam.release()
    del cam
    results['capture'] = {'output_idx': idx, 'poll_cost_ms': percentiles(costs)}

    with Capture(output_idx=idx) as cap:
        results['capture']['frame'] = [cap.width, cap.height]
        # a fast flasher for the pipeline's frame-rate ceiling
        f240 = Flasher(fps=240)
        try:
            f240.check_clock()
            results['capture']['rate_flasher_240'] = bench_capture_rate(cap, f240)
        finally:
            f240.close()
        time.sleep(0.5)
        # the application-like flasher for latency
        f = Flasher(fps=fps)
        try:
            results['clock'] = f.check_clock()
            results['capture']['rate_flasher_%d' % fps] = bench_capture_rate(cap, f)
            results['display_latency'] = bench_display_latency(cap, f, trials)
            results['input_latency'] = bench_input_latency(cap, f, act, trials)
        finally:
            f.close()

    results['cursor'] = bench_cursor(act)
    results['tick_idle_100hz'] = bench_tick(100, tick_seconds)

    if brain:
        try:
            results['brain'] = bench_brain('cuda')
            if cpu:
                results['brain_cpu'] = bench_brain('cpu', batches=(1,), steps=20)
            b = load_brain('cuda')
            if b is not None:
                import torch
                W = b.weight_matrix().detach()
                state = b.init_state(1)
                obs = {k: torch.zeros(1, d, device=b.device) for k, d in b.channel_dims.items()}

                def work():
                    nonlocal state
                    with torch.no_grad():
                        _, state, _ = b(obs, state, W)
                        torch.cuda.synchronize()
                results['tick_brain_100hz'] = bench_tick(100, tick_seconds, work)
        except Exception as e:  # the bench must always produce its table
            results['brain'] = {'available': False, 'error': f'{type(e).__name__}: {e}'}
    return results


def _fmt(p: dict, unit='ms') -> str:
    if not p or p.get('n', 0) == 0:
        return 'n/a'
    return f"p50 {p['p50']:.2f} / p90 {p['p90']:.2f} / p99 {p['p99']:.2f} / max {p['max']:.2f} {unit} (n={p['n']})"


def summarize(r: dict) -> str:
    s = r['session']
    lines = [f"ganglion bench  [{r['label']}]  {r['time']}",
             f"  session {s['id']} ({s['kind']}), console {s['console']}, screen {s['screen'][0]}x{s['screen'][1]}"]
    c = r.get('capture', {})
    if c:
        lines.append(f"  capture: DXGI output {c['output_idx']} {c['frame'][0]}x{c['frame'][1]}; poll cost {_fmt(c['poll_cost_ms'])}")
        for k, v in c.items():
            if k.startswith('rate_'):
                lines.append(f"    {k}: {v['fresh_frames_per_s']:.1f} fresh frames/s, {v['polls_per_s']:.0f} polls/s, interval {_fmt(v['interval_ms'])}")
    if 'clock' in r:
        ck = r['clock']
        lines.append(f"  cross-process clock: offset {ck['offset_ms']:.3f} ms, min round trip {ck['min_rtt_us']:.0f} us, "
                     f"{'comparable' if ck['comparable'] else 'NOT comparable, bench clock only'}")
    d = r.get('display_latency')
    if d:
        lines.append(f"  display latency ({d['trials']} trials, {d['misses']} misses):")
        lines.append(f"    cmd->flip   {_fmt(d['cmd_to_flip_ms'])}")
        lines.append(f"    flip->pixel {_fmt(d['flip_to_pixel_ms'])}")
        lines.append(f"    cmd->pixel  {_fmt(d['cmd_to_pixel_ms'])}")
    i = r.get('input_latency')
    if i:
        lines.append(f"  input latency, SendInput click ({i['trials']} trials, {i['misses']} misses):")
        lines.append(f"    click->event {_fmt(i['click_to_event_ms'])}")
        lines.append(f"    event->flip  {_fmt(i['event_to_flip_ms'])}")
        lines.append(f"    flip->pixel  {_fmt(i['flip_to_pixel_ms'])}")
        lines.append(f"    click->pixel {_fmt(i['click_to_pixel_ms'])}")
    cu = r.get('cursor')
    if cu:
        lines.append(f"  cursor: move+read {_fmt(cu['move_and_read_us'], 'us')}, max error {cu['max_error_px']} px")
    for k in ('tick_idle_100hz', 'tick_brain_100hz'):
        t = r.get(k)
        if t:
            lines.append(f"  {k}: lateness {_fmt(t)}, resyncs {t['resyncs']}")
    b = r.get('brain')
    if b:
        if b.get('available'):
            lines.append(f"  brain: {b['neurons']} neurons, {b['edges']} edges on {b['device']}, load {b['load_s']:.2f} s")
            for B, p in b['step_ms'].items():
                lines.append(f"    step B={B}: {_fmt(p)}")
        else:
            lines.append(f"  brain: not available ({b.get('error', 'no Haltere checkpoint')})")
    bc = r.get('brain_cpu')
    if bc and bc.get('available'):
        lines.append(f"  brain on CPU: step B=1 {_fmt(bc['step_ms']['1'])}")
    return '\n'.join(lines)


def save(results: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2), encoding='utf-8')
