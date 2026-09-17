from ganglion.core.output import ClickMachine
from test_runtime import Clock
import pytest


class LoggingActuator:
    """Spawn-compatible test double; no desktop APIs."""
    def __init__(self, path):
        self.path = path
        self.held = False

    def move_abs(self, x, y):
        pass

    def cursor_pos(self):
        return (1, 2)

    def button(self, name, down):
        self.held = down
        with open(self.path, "a") as log:
            log.write("down\n" if down else "up\n")

    def release_all(self):
        if self.held:
            self.button("left", False)


def accept_target(*args):
    pass


class Actuator:
    def __init__(self):
        self.held = False
        self.moves = []

    def move_abs(self, x, y):
        self.moves.append((x, y))

    def button(self, name, down):
        self.held = down

    def release_all(self):
        self.held = False


def command(clock, **kwargs):
    return {"command_id": "one", "x": 10, "y": 20, "target": {},
            "deadline": clock() + 0.5, "hold_ms": 20, **kwargs}


def test_bounded_release_without_any_further_parent_commands():
    clock, a, events = Clock(), Actuator(), []
    machine = ClickMachine(a, clock, lambda *a: None, events.append)
    machine.submit(command(clock))
    assert a.held
    clock.advance(0.021)
    machine.tick()
    assert not a.held and events[-1]["kind"] == "input_released"


def test_expired_or_occluded_target_never_presses_button():
    clock, a, events = Clock(), Actuator(), []
    def blocked(*args):
        raise ValueError("occluded")
    machine = ClickMachine(a, clock, blocked, events.append)
    machine.submit(command(clock, deadline=clock() - 1))
    assert events[-1]["kind"] == "input_cancelled" and not a.moves
    machine.submit(command(clock))
    assert events[-1]["kind"] == "input_failed" and not a.held


def test_halt_releases_immediately_and_busy_rejects_overlap():
    clock, a, events = Clock(), Actuator(), []
    machine = ClickMachine(a, clock, lambda *a: None, events.append)
    machine.submit(command(clock))
    machine.submit(command(clock, command_id="two"))
    assert events[-1]["kind"] == "input_cancelled"
    machine.halt()
    assert not a.held


@pytest.mark.parametrize("operation", ["click", "drag"])
def test_input_process_releases_after_parent_exits(tmp_path, operation):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    log = tmp_path / "input.txt"
    # Parent uses os._exit: no finally blocks, atexit handlers, or runtime cleanup.
    code = """
import functools, os, sys, time
sys.path.insert(0, sys.argv[1])
from test_output import LoggingActuator, accept_target
from ganglion.core.output import ProcessOutput
if __name__ == '__main__':
    output = ProcessOutput(_actuator_factory=functools.partial(LoggingActuator, sys.argv[2]),
                           _validate_target=accept_target)
    command = {'command_id':'crash', 'x':1, 'y':2, 'target':{},
               'deadline':time.perf_counter()+1, 'hold_ms':100,
               'drag_id':'test', 'point':(1,2), 'hold_until':time.perf_counter()+5}
    if sys.argv[3] == 'drag':
        output.begin_drag(command)
    else:
        output.submit(command)
    until = time.perf_counter()+2
    while time.perf_counter() < until:
        if any(e['kind'] in ('input_submitted', 'drag_started') for e in output.poll()):
            os._exit(0)
        time.sleep(.001)
    os._exit(2)
"""
    completed = subprocess.run([sys.executable, "-c", code, str(Path(__file__).parent), str(log), operation],
                               timeout=10, capture_output=True, env=os.environ.copy())
    assert completed.returncode == 0, completed.stderr.decode()
    until = time.perf_counter() + 2
    while time.perf_counter() < until:
        if log.exists() and log.read_text().splitlines() == ["down", "up"]:
            return
        time.sleep(0.01)
    raise AssertionError("input helper failed to release after its parent exited")


def test_hold_deadline_cannot_be_extended_after_it_expires():
    from test_drag import Plant
    c, plant, events = Clock(), Plant(), []
    machine = ClickMachine(plant, c, accept_target, events.append)
    held = command(c, point=(1, 2), drag_id="one", hold_until=c() + 10)
    machine.begin_drag(held)
    assert plant.held and machine.release_at <= c() + .1
    c.advance(.11)
    machine.pointer(held | {"command_id": "late", "point": (100, 100), "hold_until": c() + 10})
    assert not plant.held and plant.cursor == (1, 2)
    assert events[-1]["kind"] == "input_cancelled"


def test_release_failure_retains_ownership_and_prevents_new_input():
    from test_drag import Plant
    c, events = Clock(), []
    class FailedRelease(Plant):
        def release_all(self):
            raise OSError("blocked release")
    plant = FailedRelease()
    machine = ClickMachine(plant, c, accept_target, events.append)
    machine.begin_drag(command(c, point=(1, 2), drag_id="one", hold_until=c() + .1))
    assert machine.halt() is False
    assert machine.active and machine.drag_id == "one" and machine.faulted
    machine.submit(command(c, command_id="no"))
    assert events[-1]["kind"] == "input_cancelled"
