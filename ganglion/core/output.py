"""Bounded pointer execution in a separate process.

The helper owns SendInput and releases on deadline, parent pipe EOF, halt, or error.
Drag holds require continuing bounded renewals from the producer. No raw hold API is exposed.
"""
from __future__ import annotations

from collections import deque
import multiprocessing as mp
import time


class ClickMachine:
    """Small independently testable input watchdog, also used by the live helper."""
    def __init__(self, actuators, clock, validate, emit):
        self.actuators, self.clock, self.validate, self.emit = actuators, clock, validate, emit
        self.active = None
        self.release_at = 0.0
        self.drag_id = None
        self.drag_target = None
        self.last_guard = 0.0
        self.faulted = False
        self.keys = {}        # key -> (command_id, release_at)
        self.buttons = {}     # button -> (command_id, release_at) for holds without a pointer move
        self.looks = deque()  # (command_id, dx, dy, due, last) relative chunks still to send

    def event(self, kind, command_id, **data):
        self.emit({"kind": kind, "command_id": command_id, "t_mono": self.clock(), **data})

    def submit(self, command):
        cid = command["command_id"]
        if self.active is not None or self.faulted:
            self.event("input_cancelled", cid, reason="pointer_busy")
            return
        if self.clock() >= command["deadline"]:
            self.event("input_cancelled", cid, reason="deadline_expired")
            return
        try:
            self.validate(command["target"], command["x"], command["y"])
            if self.clock() >= command["deadline"]:
                self.event("input_cancelled", cid, reason="deadline_expired")
                return
            self.actuators.move_abs(command["x"], command["y"])
            # Guard again after movement, before the button event.
            self.validate(command["target"], command["x"], command["y"])
            if self.clock() >= command["deadline"]:
                self.event("input_cancelled", cid, reason="deadline_expired")
                return
            self.active = cid
            self.actuators.button("left", True)
            self.release_at = min(command["deadline"], self.clock() + command["hold_ms"] / 1000)
            self.event("input_submitted", cid, x=command["x"], y=command["y"])
        except Exception as exc:
            self.event("input_failed", cid, error=str(exc))
            self.halt()

    def tick(self):
        now = self.clock()
        if self.keys or self.buttons or self.looks:
            self._tick_extras(now)
        if self.active is not None and self.clock() >= self.release_at:
            self.halt(reason="hold_deadline_expired" if self.drag_id else "click_deadline")
        elif self.drag_id and self.clock() - self.last_guard >= .01:
            self.last_guard = self.clock()
            try:
                self.validate(self.drag_target, *self.actuators.cursor_pos())
            except Exception as exc:
                self.event("input_failed", self.active, drag_id=self.drag_id, error=str(exc))
                self.faulted = True
                self.halt(reason="target_invalid")

    def _tick_extras(self, now):
        expired = {}
        for key, (cid, until) in list(self.keys.items()):
            if now >= until:
                try:
                    self.actuators.key(key, False)
                except Exception as exc:
                    self.event("input_failed", cid, error=str(exc))
                    self.halt()
                    return
                del self.keys[key]
                expired.setdefault(cid, []).append(key)
        for cid, keys in expired.items():
            self.event("keys_released", cid, keys=keys, reason="hold_deadline")
        for button, (cid, until) in list(self.buttons.items()):
            if now >= until:
                try:
                    self.actuators.button(button, False)
                except Exception as exc:
                    self.event("input_failed", cid, error=str(exc))
                    self.halt()
                    return
                del self.buttons[button]
                self.event("input_released", cid, button=button, reason="hold_deadline")
        while self.looks and self.looks[0][3] <= now:
            cid, dx, dy, _, last = self.looks.popleft()
            try:
                self.actuators.move_rel(dx, dy)
            except Exception as exc:
                self.event("input_failed", cid, error=str(exc))
                self.halt()
                return
            if last:
                self.event("look_done", cid)

    def key(self, command):
        """Hold keys for a bounded time; re-issuing the same keys extends the hold, never past 1 s."""
        cid = command["command_id"]
        if self.faulted:
            self.event("input_cancelled", cid, reason="input_faulted")
            return
        if self.clock() >= command["deadline"]:
            self.event("input_cancelled", cid, reason="deadline_expired")
            return
        try:
            self.validate(command["target"])
            until = min(command["hold_until"], self.clock() + 1.0)
            if self.clock() >= until:
                self.event("input_cancelled", cid, reason="deadline_expired")
                return
            extended, pressed = [], []
            for key in command["keys"]:
                if key in self.keys:
                    old_cid = self.keys[key][0]
                    self.keys[key] = (cid if old_cid == cid else old_cid, until)
                    extended.append(key)
                else:
                    self.actuators.key(key, True)
                    self.keys[key] = (cid, until)
                    pressed.append(key)
            self.event("keys_held", cid, keys=pressed, extended=extended, hold_until=until)
        except Exception as exc:
            self.event("input_failed", cid, error=str(exc))
            self.halt()

    def look(self, command):
        """Relative mouse motion in counts, as one delta or spread over chunks 10 ms apart."""
        cid = command["command_id"]
        if self.faulted:
            self.event("input_cancelled", cid, reason="input_faulted")
            return
        if self.clock() >= command["deadline"]:
            self.event("input_cancelled", cid, reason="deadline_expired")
            return
        try:
            self.validate(command["target"])
            if self.clock() >= command["deadline"]:
                self.event("input_cancelled", cid, reason="deadline_expired")
                return
            chunks = max(1, int(command.get("chunks", 1)))
            dx, dy = int(command["dx"]), int(command["dy"])
            sent_x = sent_y = 0
            now = self.clock()
            for i in range(1, chunks + 1):
                px, py = round(dx * i / chunks), round(dy * i / chunks)
                step = (px - sent_x, py - sent_y)
                sent_x, sent_y = px, py
                if i == 1:
                    self.actuators.move_rel(*step)
                    if chunks == 1:
                        self.event("look_done", cid, dx=dx, dy=dy)
                else:
                    self.looks.append((cid, step[0], step[1], now + (i - 1) * .01, i == chunks))
            if chunks > 1:
                self.event("look_submitted", cid, dx=dx, dy=dy, chunks=chunks)
        except Exception as exc:
            self.event("input_failed", cid, error=str(exc))
            self.halt()

    def release(self, command):
        """Let named keys go now (a move program ending) instead of at their hold deadline."""
        cid = command["command_id"]
        released = []
        try:
            for key in command["keys"]:
                if key in self.keys:
                    self.actuators.key(key, False)
                    del self.keys[key]
                    released.append(key)
        except Exception as exc:
            self.event("input_failed", cid, error=str(exc))
            self.halt()
            return
        self.event("keys_released", cid, keys=released, reason="requested")

    def button(self, command):
        """Hold a mouse button without moving the pointer; released on deadline or halt."""
        cid = command["command_id"]
        button = command["button"]
        if self.faulted or button in self.buttons or (button == "left" and self.active is not None):
            self.event("input_cancelled", cid, reason="pointer_busy")
            return
        if self.clock() >= command["deadline"]:
            self.event("input_cancelled", cid, reason="deadline_expired")
            return
        try:
            self.validate(command["target"])
            until = min(command["hold_until"], self.clock() + 1.0)
            if self.clock() >= min(command["deadline"], until):
                self.event("input_cancelled", cid, reason="deadline_expired")
                return
            self.actuators.button(button, True)
            self.buttons[button] = (cid, until)
            self.event("input_submitted", cid, button=button, hold_until=until)
        except Exception as exc:
            self.event("input_failed", cid, error=str(exc))
            self.halt()

    def begin_drag(self, command):
        cid = command["command_id"]
        if self.active is not None or self.faulted:
            self.event("input_cancelled", cid, reason="pointer_busy")
            return
        if self.clock() >= command["deadline"]:
            self.event("input_cancelled", cid, reason="deadline_expired")
            return
        try:
            point = command["point"]
            self.validate(command["target"], *point)
            if self.clock() >= min(command["deadline"], command["hold_until"]):
                self.event("input_cancelled", cid, reason="deadline_expired")
                return
            self.actuators.move_abs(*point)
            self.validate(command["target"], *point)
            until = min(command["hold_until"], self.clock() + .1)
            if self.clock() >= min(command["deadline"], until):
                self.event("input_cancelled", cid, reason="deadline_expired")
                return
            self.active, self.drag_id, self.drag_target = cid, command["drag_id"], command["target"]
            self.release_at = until
            self.actuators.button("left", True)
            self.event("drag_started", cid, drag_id=self.drag_id, cursor=list(self.actuators.cursor_pos()),
                       hold_until=until)
        except Exception as exc:
            self.faulted = True
            self.event("input_failed", cid, error=str(exc))
            self.halt()

    def end_drag(self, command):
        if not self.drag_id or self.drag_id != command["drag_id"]:
            self.event("input_cancelled", command["command_id"], reason="drag_not_held")
            return
        # Releasing must remain possible after expiry, focus loss, or a stale observation.
        self.halt(command_id=command["command_id"], reason="requested")

    def pointer(self, command):
        """Sample the OS cursor, optionally after a bounded move (never assume arrival)."""
        cid = command["command_id"]
        requested_drag = command.get("drag_id")
        if requested_drag and (requested_drag != self.drag_id or self.clock() >= self.release_at):
            self.tick()
            self.event("input_cancelled", cid, reason="drag_not_held")
            return
        if requested_drag and self.clock() >= command["hold_until"]:
            self.halt(reason="hold_deadline_expired")
            self.event("input_cancelled", cid, reason="hold_deadline_expired")
            return
        if self.faulted or (self.active is not None and not requested_drag):
            self.event("input_cancelled", cid, reason="pointer_busy")
            return
        if self.clock() >= command["deadline"]:
            self.event("input_cancelled", cid, reason="deadline_expired")
            return
        try:
            point = command.get("point")
            self.validate(command["target"], *(point or ()))
            if self.clock() >= command["deadline"]:
                self.event("input_cancelled", cid, reason="deadline_expired")
                return
            if point is not None:
                self.actuators.move_abs(*point)
            if requested_drag:
                # Parent-provided bounds can shorten the 100 ms dead-man, never lengthen it.
                self.release_at = min(command["hold_until"], self.clock() + .1)
                if self.clock() >= self.release_at:
                    self.halt(reason="hold_deadline_expired")
                    self.event("input_cancelled", cid, reason="hold_deadline_expired")
                    return
            # SendInput may still be queued. This is the observed position, not its argument.
            cursor = self.actuators.cursor_pos()
            self.event("pointer_feedback", cid, cursor=list(cursor), submitted_point=point)
        except Exception as exc:
            self.event("input_failed", cid, error=str(exc))
            self.halt()

    def halt(self, *, command_id=None, reason="halt"):
        cid, drag_id = command_id or self.active, self.drag_id
        try:
            self.actuators.release_all()
        except Exception as exc:
            self.faulted = True
            self.release_at = self.clock() + .01  # retain ownership and retry release
            self.event("input_failed", cid or "release", error=f"release failed: {exc}")
            return False
        self.active, self.drag_id, self.drag_target = None, None, None
        if cid is not None:
            self.event("input_released", cid, drag_id=drag_id, reason=reason)
        released = {}
        for key, (key_cid, _) in self.keys.items():
            released.setdefault(key_cid, []).append(key)
        self.keys.clear()
        for key_cid, keys in released.items():
            self.event("keys_released", key_cid, keys=keys, reason=reason)
        for button, (button_cid, _) in list(self.buttons.items()):
            self.event("input_released", button_cid, button=button, reason=reason)
        self.buttons.clear()
        for look_cid in {item[0] for item in self.looks}:
            self.event("input_cancelled", look_cid, reason=reason)
        self.looks.clear()
        return True


def _worker(connection, actuator_factory=None, validate_target=None):
    if actuator_factory is None:
        from .actuators import Actuators
        actuator_factory = Actuators
    if validate_target is None:
        from .window import validate
        validate_target = validate
    from .clock import TimerPeriod
    machine = ClickMachine(actuator_factory(), time.perf_counter, validate_target, connection.send)
    try:
        with TimerPeriod(1):
            connection.send({"kind": "ready"})
            while True:
                machine.tick()
                if not connection.poll(0.002):
                    continue
                op, payload = connection.recv()
                if op == "close":
                    return
                if op == "halt":
                    released = machine.halt()
                    if payload and released:
                        machine.event("output_halted", payload)
                elif op == "click":
                    machine.submit(payload)
                elif op == "pointer":
                    machine.pointer(payload)
                elif op == "begin_drag":
                    machine.begin_drag(payload)
                elif op == "end_drag":
                    machine.end_drag(payload)
                elif op in ("key", "look", "button", "release"):
                    getattr(machine, op)(payload)
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        # Release first even when emitting back to a dead parent fails.
        try:
            machine.halt()
        finally:
            connection.close()


class ProcessOutput:
    def __init__(self, *, _actuator_factory=None, _validate_target=None):
        ctx = mp.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, _actuator_factory, _validate_target),
                                   name="ganglion-input")
        self.process.start()
        child.close()
        if not self.connection.poll(10) or self.connection.recv().get("kind") != "ready":
            self.close()
            raise RuntimeError("input helper did not become ready")

    def healthy(self):
        return self.process.is_alive()

    def submit(self, command):
        self.connection.send(("click", command))

    def pointer(self, command):
        self.connection.send(("pointer", command))

    def begin_drag(self, command):
        self.connection.send(("begin_drag", command))

    def end_drag(self, command):
        self.connection.send(("end_drag", command))

    def key(self, command):
        self.connection.send(("key", command))

    def look(self, command):
        self.connection.send(("look", command))

    def button(self, command):
        self.connection.send(("button", command))

    def release(self, command):
        self.connection.send(("release", command))

    def halt(self, barrier_id=None):
        if self.healthy():
            self.connection.send(("halt", barrier_id))

    def poll(self):
        events = []
        try:
            while self.connection.poll():
                events.append(self.connection.recv())
        except (EOFError, OSError):
            pass
        return events

    def close(self):
        try:
            if self.healthy():
                self.connection.send(("close", None))
        except (BrokenPipeError, OSError):
            pass
        self.connection.close()
        self.process.join(2)
        # Do not terminate a possibly input-owning process before its bounded release.
        if self.process.is_alive():
            raise RuntimeError("input helper failed to exit; inspect its process before restarting")


class MemoryOutput:
    """Headless output for tests and the synthetic Arena; never imports desktop APIs."""
    def __init__(self, clock, on_click=None, on_look=None):
        self.clock, self.on_click, self.on_look = clock, on_click, on_look
        self.events, self.commands = [], []
        self.alive = True
        self.cursor = (0, 0)

    def healthy(self):
        return self.alive

    def submit(self, command):
        self.commands.append(command)
        if self.clock() >= command["deadline"]:
            self.events.append({"kind": "input_cancelled", "command_id": command["command_id"],
                                "t_mono": self.clock(), "reason": "deadline_expired"})
            return
        self.events.append({"kind": "input_submitted", "command_id": command["command_id"],
                            "t_mono": self.clock()})
        if self.on_click:
            self.on_click(command["x"], command["y"])
        self.events.append({"kind": "input_released", "command_id": command["command_id"],
                            "t_mono": self.clock()})

    def poll(self):
        events, self.events = self.events, []
        return events

    def pointer(self, command):
        self.commands.append(command)
        if self.clock() >= command["deadline"]:
            self.events.append({"kind": "input_cancelled", "command_id": command["command_id"],
                                "t_mono": self.clock(), "reason": "deadline_expired"})
            return
        if command.get("point") is not None:
            self.cursor = tuple(command["point"])
        self.events.append({"kind": "pointer_feedback", "command_id": command["command_id"],
                            "t_mono": self.clock(), "cursor": list(self.cursor),
                            "submitted_point": command.get("point")})

    def key(self, command):
        self.commands.append(command)
        if self.clock() >= command["deadline"]:
            self.events.append({"kind": "input_cancelled", "command_id": command["command_id"],
                                "t_mono": self.clock(), "reason": "deadline_expired"})
            return
        self.events.append({"kind": "keys_held", "command_id": command["command_id"], "t_mono": self.clock(),
                            "keys": list(command["keys"]), "extended": [], "hold_until": command["hold_until"]})
        self.events.append({"kind": "keys_released", "command_id": command["command_id"], "t_mono": self.clock(),
                            "keys": list(command["keys"]), "reason": "hold_deadline"})

    def look(self, command):
        self.commands.append(command)
        if self.clock() >= command["deadline"]:
            self.events.append({"kind": "input_cancelled", "command_id": command["command_id"],
                                "t_mono": self.clock(), "reason": "deadline_expired"})
            return
        if self.on_look:
            self.on_look(command["dx"], command["dy"])
        self.events.append({"kind": "look_done", "command_id": command["command_id"], "t_mono": self.clock(),
                            "dx": command["dx"], "dy": command["dy"]})

    def release(self, command):
        self.commands.append(command)
        self.events.append({"kind": "keys_released", "command_id": command["command_id"], "t_mono": self.clock(),
                            "keys": list(command["keys"]), "reason": "requested"})

    def button(self, command):
        self.commands.append(command)
        if self.clock() >= command["deadline"]:
            self.events.append({"kind": "input_cancelled", "command_id": command["command_id"],
                                "t_mono": self.clock(), "reason": "deadline_expired"})
            return
        self.events.append({"kind": "input_submitted", "command_id": command["command_id"], "t_mono": self.clock(),
                            "button": command["button"]})
        if self.on_click:
            self.on_click(None, None)
        self.events.append({"kind": "input_released", "command_id": command["command_id"], "t_mono": self.clock(),
                            "button": command["button"], "reason": "hold_deadline"})

    def halt(self, barrier_id=None):
        if barrier_id:
            self.events.append({"kind": "output_halted", "command_id": barrier_id,
                                "t_mono": self.clock()})

    def close(self):
        self.alive = False


class SimulatedOutput:
    """Run the same input watchdog against an injected simulated actuator."""
    def __init__(self, clock, actuator, validate=lambda *args: None):
        self.events, self.commands = [], []
        self.machine = ClickMachine(actuator, clock, validate, self.events.append)
        self.alive = True

    def healthy(self):
        return self.alive

    def _send(self, op, command):
        self.machine.tick()
        self.commands.append(command | {"op": op})
        getattr(self.machine, op)(command)

    def submit(self, command):
        self._send("submit", command)

    def pointer(self, command):
        self._send("pointer", command)

    def begin_drag(self, command):
        self._send("begin_drag", command)

    def end_drag(self, command):
        self._send("end_drag", command)

    def key(self, command):
        self._send("key", command)

    def look(self, command):
        self._send("look", command)

    def button(self, command):
        self._send("button", command)

    def release(self, command):
        self._send("release", command)

    def poll(self):
        self.machine.tick()
        events, self.events = self.events, []
        self.machine.emit = self.events.append
        return events

    def halt(self, barrier_id=None):
        if self.machine.halt() and barrier_id:
            self.machine.event("output_halted", barrier_id)

    def close(self):
        self.halt()
        self.alive = False
