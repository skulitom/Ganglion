"""The `ganglion` command."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ganglion', description='A fly-brain reflex layer for LLM agents.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('doctor', help='what is installed, which session this is, what would block the fast loop')
    p.add_argument('--json', action='store_true')
    p.add_argument('--quick', action='store_true', help='skip the capture probe')

    p = sub.add_parser('bench', help='capture rate/latency, input-to-pixel latency, brain step time, tick jitter')
    p.add_argument('--json', metavar='PATH', help='write the results as JSON')
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--fps', type=int, default=60, help='frame rate of the application-like flasher')
    p.add_argument('--no-brain', action='store_true')
    p.add_argument('--cpu', action='store_true', help='also time one brain step on the CPU (slow)')
    p.add_argument('--label', help='name for this run (default: the session kind)')
    p.add_argument('--tick-seconds', type=float, default=3.0)

    p = sub.add_parser('flasher', help='run the timing-instrumented flasher window by hand')
    p.add_argument('args', nargs=argparse.REMAINDER)

    p = sub.add_parser('core', help='run the resident reflex service in this session')
    p.add_argument('--endpoint', required=True, help='write private connection credentials to this file')
    p.add_argument('--window', type=int, help='target HWND for live capture and input')
    p.add_argument('--pid', type=int, help='select the single visible top-level window of this process instead of --window')
    p.add_argument('--title', help='select the single visible top-level window whose title contains this text')
    p.add_argument('--synthetic', action='store_true', help='headless Arena; no desktop input')
    p.add_argument('--seconds', type=float, default=0, help='stop after this many seconds; 0 runs until interrupted')
    p.add_argument('--shadow-checkpoint', help='optional Haltere connectome checkpoint; predictions have no control authority')
    p.add_argument('--log', metavar='PATH', help='append stdout/stderr to this file (for windowless launches such as pythonw)')

    p = sub.add_parser('mcp', help='MCP stdio bridge to a running core')
    p.add_argument('--endpoint', required=True)
    p.add_argument('--observer', action='store_true')
    p.add_argument('--client-id')

    p = sub.add_parser('demo', help='Gate B: Arena pixels to click reflex through MCP')
    p.add_argument('--synthetic', action='store_true', help='headless integration, never touches the desktop')
    p.add_argument('--seconds', type=float, default=6)
    p.add_argument('--json', metavar='PATH')

    p = sub.add_parser('reach-demo', help='Compare periodic input and cursor-feedback reach through MCP')
    p.add_argument('--environment', choices=['synthetic', 'arena', 'browser'], default='synthetic')
    p.add_argument('--trials', type=int, default=4, help='paired seeds per policy, 1–8')
    p.add_argument('--json', metavar='PATH')
    p.add_argument('--shadow-checkpoint', help='compare real connectome predictions with live reach decisions')
    p.add_argument('--controller', choices=['deterministic', 'connectome'], default='deterministic',
                   help='connectome: its proposals drive the cursor when they pass the supervising envelope')

    sub.add_parser('version')
    p = sub.add_parser('drag-demo', help='Static-screen reach and bounded drag through MCP')
    p.add_argument('--environment', choices=['synthetic', 'arena', 'browser'], default='synthetic')
    p.add_argument('--trials', type=int, default=2)
    p.add_argument('--json', metavar='PATH')
    p.add_argument('--shadow-checkpoint', help='optional Haltere connectome checkpoint')
    p.add_argument('--controller', choices=['deterministic', 'connectome'], default='deterministic')

    a = ap.parse_args(argv)
    if a.cmd == 'version':
        from . import __version__
        print(__version__)
        return 0
    if a.cmd == 'doctor':
        from . import doctor
        checks = doctor.run(quick=a.quick)
        print(json.dumps(doctor.to_json(checks), indent=2) if a.json else doctor.format_checks(checks))
        return 1 if any(c.state == 'FAIL' for c in checks) else 0
    if a.cmd == 'bench':
        from .bench import run as bench
        results = bench.run(trials=a.trials, fps=a.fps, brain=not a.no_brain, cpu=a.cpu, label=a.label,
                            tick_seconds=a.tick_seconds)
        print(bench.summarize(results))
        if a.json:
            bench.save(results, a.json)
            print(f'saved {a.json}')
        return 0
    if a.cmd == 'flasher':
        from .arena import flasher
        return flasher.main(a.args)
    if a.cmd == 'core':
        from .core.runner import run_core
        if a.log:
            stream = open(a.log, 'a', buffering=1, encoding='utf-8')
            sys.stdout = sys.stderr = stream
        return run_core(a.endpoint, hwnd=a.window, pid=a.pid, title=a.title, synthetic=a.synthetic,
                        seconds=a.seconds, shadow_checkpoint=a.shadow_checkpoint)
    if a.cmd == 'mcp':
        from .mcp.server import build
        build(a.endpoint, observer=a.observer, client_id=a.client_id).run()
        return 0
    if a.cmd == 'demo':
        from .arena.demo import run
        result = run(synthetic=a.synthetic, seconds=a.seconds, path=a.json)
        print(json.dumps({k: v for k, v in result.items() if k not in ('events', 'truth')}, indent=2))
        return 0 if result['passed'] else 1
    if a.cmd == 'reach-demo':
        from .arena.reach_demo import run
        result = run(environment=a.environment, trials=a.trials, path=a.json,
                     shadow_checkpoint=a.shadow_checkpoint, controller=a.controller)
        print(json.dumps({k: v for k, v in result.items() if k not in ('events', 'truth', 'trials')}, indent=2))
        return 0 if result['passed'] else 1
    if a.cmd == 'drag-demo':
        from .arena.drag_demo import run
        result = run(environment=a.environment, trials=a.trials, path=a.json,
                     shadow_checkpoint=a.shadow_checkpoint, controller=a.controller)
        print(json.dumps({k: v for k, v in result.items() if k not in ('events', 'truth', 'trials')}, indent=2))
        return 0 if result['passed'] else 1
    return 2


if __name__ == '__main__':
    sys.exit(main())
