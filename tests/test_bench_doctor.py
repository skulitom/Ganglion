import sys

import pytest

from ganglion.arena import flasher
from ganglion.bench import run as bench
from ganglion import doctor

win = pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')


def test_flasher_protocol_parse():
    assert flasher.parse_reply('FLIP 123 click 100 1 1 -1\n') == ('FLIP', [123, 'click', 100, 1, 1, -1])
    assert flasher.parse_reply('READY 10 20 320 240 60') == ('READY', [10, 20, 320, 240, 60])
    assert flasher.parse_reply('') == ('', [])


def test_flasher_command_line():
    cmd = flasher.command_line('py', 9700, 9701, 1, 2, 3, 4, 60)
    assert cmd[:3] == ['py', '-m', 'ganglion.arena.flasher'] and '--fps' in cmd and cmd[cmd.index('--fps') + 1] == '60'


def test_summarize_handles_partial_results():
    r = {'label': 'x', 'time': 't', 'session': {'id': 1, 'kind': 'console', 'console': 1, 'screen': [1, 1]},
         'brain': {'available': False}, 'tick_idle_100hz': {'p50': 0.1, 'p90': 0.2, 'p99': 0.3, 'max': 1.0, 'n': 10, 'resyncs': 0, 'hz': 100}}
    text = bench.summarize(r)
    assert 'brain: not available' in text and 'tick_idle_100hz' in text


@win
def test_doctor_runs_quick():
    checks = doctor.run(quick=True)
    names = {c.name for c in checks}
    assert {'python', 'session', 'capture', 'torch'} <= names
    assert all(c.state in ('OK', 'WARN', 'FAIL', 'SKIP') for c in checks)
    assert doctor.format_checks(checks).endswith(('OK', 'WARN', 'FAIL'))
