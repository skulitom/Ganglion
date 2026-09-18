"""Deterministic state machine; no desktop APIs, networking, or sleeps.

The runner supplies timestamped observations and polls the independent input worker.
Tests use a fake clock and output to exercise the same state transitions.
"""
from __future__ import annotations

from math import hypot
from threading import RLock
from uuid import uuid4

from .errors import RuntimeErrorWithCode  # noqa: F401  (re-exported for callers and tests)
from .ledger import Ledger
from .perception import ChangeSense, Reflex, Watch, new_watch, observe
from .schema import ArmSpec, WatchSpec, IntentSpec, InputSpec
from .reach import Reach, step, supervise
from .drag import Drag, drive_drag, drag_event
from .aim import Align, drive_align, align_event
from .locomotion import Move, drive_move, finish_move


FLOW_MAX_AGE = .15      # seconds a flow summary stays fed to the model after its capture

class Runtime:
    def __init__(self, clock, output, *, ledger=None, session_id=0, shadow_predictor=None, shadow_factory=None,
                 lptc_feed=False):
        self.clock, self.output = clock, output
        self.lptc_feed = lptc_feed              # hand the newest wide-field flow to the model's lptc channel
        self.flow = None                        # newest ego-motion summary from a flow watch
        self.ledger = ledger or Ledger()
        self.session_id = session_id
        self.lock = RLock()
        self.state, self.reason = "ready", None
        self.owner = None
        self.lease_id = None
        self.expires = 0.0
        self.watches: dict[str, Watch] = {}
        self.reflexes: dict[str, Reflex] = {}
        self.pending: dict[str, dict] = {}      # pointer, button and look commands in flight
        self.key_pending: dict[str, dict] = {}  # key holds are a separate actuator
        self.motion_until = 0.0                 # own commands move the view or the player until then
        self.intent: Reach | None = None
        self.locomotion: Move | None = None
        self.frame = None
        self.frame_seq = 0
        self.frame_time = 0.0
        self.frame_source = None
        self.source_counts = {}
        self.frame_started = 0.0
        self.output_fault = False
        self.layout = None
        self.layout_rev = 0
        self.dropped_frames = 0
        self.change = None            # mean grey change of the upper frame against the previous one
        self.change_sense = ChangeSense()
        self.ledger.append("core_ready", clock(), critical=True)
        self.shadow = None
        if shadow_factory is not None:
            # A picklable callable that builds the predictor in a child process, away from this interpreter.
            from ganglion.brain.shadow_process import ShadowProcess
            self.shadow = ShadowProcess(shadow_factory, self.ledger, clock=clock)
        elif shadow_predictor is not None:
            from ganglion.brain.shadow import ShadowWorker
            self.shadow = ShadowWorker(shadow_predictor, self.ledger, clock=clock)

    def _error(self, code, message):
        raise RuntimeErrorWithCode(code, message)

    def note_self_motion(self, until):
        self.motion_until = max(self.motion_until, until)

    def client_centre(self):
        x, y, w, h = self.layout["rect"]
        return x + w // 2, y + h // 2

    def _require_owner(self, client):
        self.tick(drive=False)
        if self.owner != client or self.expires <= self.clock():
            self._error("lease_required", "Claim control, then explicitly renew before expiry.")
        if self.state in ("halted", "degraded"):
            self._error("not_ready", "Claim a fresh lease and rebind watches after the fault.")

    def claim(self, client, seconds):
        with self.lock:
            self.tick()
            if self.owner is not None and self.owner != client:
                self._error("lease_busy", "Another client owns control; wait for expiry or release.")
            if self.owner == client:
                return self.renew(client, seconds)
            if not self.output.healthy() or self.output_fault:
                self._error("output_unavailable", "Restart the core: the input worker is unavailable.")
            self.owner, self.expires = client, self.clock() + seconds
            self.lease_id = uuid4().hex
            self.state, self.reason = "ready", None
            self.ledger.append("lease_claimed", self.clock(), critical=True, client_id=client)
            return self.lease()

    def renew(self, client, seconds):
        with self.lock:
            self._require_owner(client)
            self.expires = self.clock() + seconds
            return self.lease()

    def lease(self):
        return {"owner": self.owner, "lease_id": self.lease_id,
                "expires_mono": self.expires, "remaining_seconds": max(0, self.expires - self.clock())}

    def halt(self, reason="requested", *, degraded=False):
        with self.lock:
            if self.shadow:
                self.shadow.invalidate()
            self._finish_intent("cancelled", reason, stop=False)
            if self.locomotion and self.locomotion.active:
                self.locomotion.phase, self.locomotion.reason = "cancelled", reason
                self.ledger.append("intent_cancelled", self.clock(), critical=True, **self.locomotion.status())
            self.reflexes.clear()
            self.watches.clear()
            self.flow = None
            self.owner, self.lease_id, self.expires = None, None, 0.0
            self.state = "degraded" if degraded else "halted"
            self.reason = reason
            try:
                self.output.halt()
            except Exception as exc:
                self.state, self.reason = "degraded", f"release_failed: {exc}"
                self.output_fault = True
            self.ledger.append("halted", self.clock(), critical=True, reason=self.reason,
                               state=self.state)
            return {"state": self.state, "reason": self.reason}

    def tick(self, *, drive=True):
        with self.lock:
            now = self.clock()
            if self.owner is not None and now >= self.expires:
                self.halt("lease_expired")
            for rid, reflex in list(self.reflexes.items()):
                if now >= reflex.expires:
                    del self.reflexes[rid]
                    self.ledger.append("reflex_expired", now, critical=True, reflex_id=rid)
            for event in self.output.poll():
                command_id = event["command_id"]
                details = self.pending.get(command_id) or self.key_pending.get(command_id, {})
                self.ledger.append(event["kind"], event["t_mono"],
                                   critical=event["kind"] not in ("pointer_feedback", "look_done"),
                                   **details, **{k: v for k, v in event.items()
                                                 if k not in ("kind", "t_mono")})
                if event["kind"] in ("input_released", "input_failed", "input_cancelled",
                                     "pointer_feedback", "output_halted", "drag_started", "look_done"):
                    self.pending.pop(command_id, None)
                if event["kind"] in ("keys_released", "input_failed", "input_cancelled"):
                    self.key_pending.pop(command_id, None)
                intent = self.intent
                if isinstance(intent, Drag) and (event.get("drag_id") == intent.id or details.get("intent_id") == intent.id):
                    drag_event(self, intent, event)
                if isinstance(intent, Align):
                    align_event(self, intent, event, details)
                elif intent and intent.active and details.get("intent_id") == intent.id:
                    if event["kind"] == "pointer_feedback":
                        intent.cursor = tuple(event["cursor"])
                        intent.cursor_time = event["t_mono"]
                    elif event["kind"] == "input_cancelled":
                        if event.get("reason") == "deadline_expired":
                            # The helper confirms no button action occurred. Re-observe and retry;
                            # never replay a possibly submitted click or a lost hold.
                            intent.cursor = None
                            intent.settled_since = None
                            if isinstance(intent, Drag) and intent.stage == "pressing":
                                intent.stage = "approaching"
                            elif not isinstance(intent, Drag):
                                intent.phase = "running"
                        else:
                            self._finish_intent("failed", event.get("reason", "input_cancelled"))
                    elif event["kind"] == "input_released" and intent.phase == "clicking":
                        self._finish_intent("completed", "cursor_confirmed_click_released", stop=False)
                if event["kind"] == "input_failed":
                    self.output_fault = True
                    self.halt(event.get("error", "input_failed"), degraded=True)
            if self.owner is not None and not self.output.healthy():
                self.halt("input_worker_lost", degraded=True)
            if drive:
                self._tick_intent(now)

    def start_intent(self, client, spec: IntentSpec):
        with self.lock:
            self._require_owner(client)
            if spec.program == "move":
                if spec.until and spec.until.watch_id not in self.watches:
                    self._error("unknown_watch", "Teach the until watch before starting the move.")
                if self.locomotion and self.locomotion.active:
                    self._error("keys_busy", "Cancel the active move before starting another.")
                now = self.clock()
                self.locomotion = Move(uuid4().hex, spec, now, now + spec.timeout_seconds)
                self.state = "running"
                self.ledger.append("intent_started", now, critical=True, **self.locomotion.status())
                drive_move(self, self.locomotion, now)
                return self.locomotion.status()
            if spec.watch_id not in self.watches:
                self._error("unknown_watch", "Create a watch before reaching it.")
            if spec.program == "drag":
                for wid in (spec.destination_watch_id, spec.until.watch_id if spec.until else None):
                    if wid is not None and wid not in self.watches:
                        self._error("unknown_watch", "Teach all drag watches before starting the intent.")
                if spec.destination:
                    x, y, w, h = self.layout["rect"]
                    px, py = spec.destination
                    if not (x <= px < x + w and y <= py < y + h
                            and px < self.frame.shape[1] and py < self.frame.shape[0]):
                        self._error("point_outside_target", "Drag destination must be inside the bound client area.")
            if self.pending or (self.intent and self.intent.active):
                self._error("pointer_busy", "Cancel the active intent and wait for pending output to drain.")
            if spec.controller == "connectome" and self.shadow is None:
                self._error("controller_unavailable",
                            "Start the core with --shadow-checkpoint to grant the connectome supervised authority.")
            if spec.program == "align" and spec.point is not None:
                x, y, w, h = self.layout["rect"]
                if not (x <= spec.point[0] < x + w and y <= spec.point[1] < y + h):
                    self._error("point_outside_target", "Align point must be inside the bound client area.")
            now = self.clock()
            factory = Drag if spec.program == "drag" else Align if spec.program == "align" else Reach
            self.intent = factory(uuid4().hex, spec, now, now + spec.timeout_seconds)
            self.state = "running"
            self.ledger.append("intent_started", now, critical=True, **self.intent.status())
            return self.intent.status()

    def cancel(self, client, ident):
        with self.lock:
            self._require_owner(client)
            if self.locomotion and self.locomotion.id == ident:
                finish_move(self, self.locomotion, "cancelled", "requested")
                return self.locomotion.status()
            if not self.intent or self.intent.id != ident:
                self._error("unknown_intent", "This intent is no longer retained by the runtime.")
            self._finish_intent("cancelled", "requested")
            return self.intent.status()

    def _finish_intent(self, phase, reason, *, stop=True):
        intent = self.intent
        if not intent or not intent.active:
            return
        if self.shadow:
            self.shadow.invalidate()
        intent.phase, intent.reason = phase, reason
        self.ledger.append("intent_" + phase, self.clock(), critical=True, **intent.status())
        cleanup = getattr(intent, "cleanup_watch", None)
        if cleanup and cleanup in self.watches:
            del self.watches[cleanup]
            self.reflexes = {k: v for k, v in self.reflexes.items() if v.spec.watch_id != cleanup}
            self.ledger.append("watch_removed", self.clock(), critical=True, id=cleanup, reason="intent_ended")
        if stop:
            # FIFO barrier: a new program cannot share the pointer with queued old output.
            # One already submitted command may execute before cancellation is acknowledged.
            cid = uuid4().hex
            self.pending[cid] = {"intent_id": intent.id}
            try:
                self.output.halt(cid)
            except Exception as exc:
                self.pending.pop(cid, None)
                self.halt(f"cancel_failed: {exc}", degraded=True)

    def _tick_intent(self, now):
        if self.locomotion and self.locomotion.active and self.owner is not None:
            drive_move(self, self.locomotion, now)
        intent = self.intent
        if not intent or not intent.active or self.owner is None:
            return
        if now >= intent.expires:
            self._finish_intent("failed", "timeout")
            return
        if isinstance(intent, Drag):
            drive_drag(self, intent, now)
            return
        if isinstance(intent, Align):
            drive_align(self, intent, now)
            return
        if intent.phase == "clicking":
            return
        watch = self.watches.get(intent.spec.watch_id)
        if watch is None:
            self._finish_intent("failed", "watch_removed")
            return
        # Wait briefly for the first post-teaching observation; never act on old evidence.
        if watch.observation_id == 0:
            if now - intent.started >= 0.25:
                self._finish_intent("failed", "observation_stale")
            return
        if not watch.present:
            self._finish_intent("failed", "target_lost")
            return
        age = now - watch.sample_started
        if age > 0.04:  # reserve transport time inside the unchanged 50 ms admission deadline
            if age >= 0.25:
                intent.settled_since = None
                self._finish_intent("failed", "observation_stale")
            return
        if self.pending or now - intent.last_tick < 0.01:
            return
        dt = min(0.02, now - intent.last_tick)
        intent.last_tick = now
        detection = watch.detection
        goal = (detection["x"], detection["y"])
        point = None
        click = False
        if intent.cursor is not None and now - intent.cursor_time <= 0.05:
            intent.error_px = hypot(goal[0] - intent.cursor[0], goal[1] - intent.cursor[1])
            if intent.cursor_time > intent.consumed_cursor_time:
                intent.consumed_cursor_time = intent.cursor_time
                if intent.error_px <= intent.spec.tolerance_px:
                    if intent.settled_since is None:
                        intent.settled_since = intent.cursor_time
                    if intent.cursor_time - intent.settled_since >= intent.spec.settle_ms / 1000:
                        if not intent.spec.click:
                            self._finish_intent("completed", "cursor_confirmed", stop=False)
                            return
                        click = True
                else:
                    intent.settled_since = None
            point = self._pointer_step(intent, goal, dt, now, watch)
        else:
            intent.settled_since = None
        cid = uuid4().hex
        self.pending[cid] = {"intent_id": intent.id, "watch_id": intent.spec.watch_id,
                             "observation_id": watch.observation_id, "captured_mono": watch.captured,
                             "percept_ready_mono": now, "goal": list(goal), "controller": intent.last_controller}
        self.pending[cid].update(capture_source=watch.source, sample_started_mono=watch.sample_started)
        command = {"command_id": cid, "target": self.layout,
                   "deadline": min(self.expires, intent.expires, watch.sample_started + 0.05)}
        try:
            if click:
                # Click at the measured cursor; do not teleport to a newer target position.
                self.output.submit(command | {"x": intent.cursor[0], "y": intent.cursor[1], "hold_ms": 20})
                watch.last_command = cid
                intent.phase = "clicking"
            else:
                self.output.pointer(command | {"point": point})
            intent.commands += 1
        except Exception as exc:
            self.pending.pop(cid, None)
            self.halt(f"submit_failed: {exc}", degraded=True)

    def _pointer_step(self, intent, goal, dt, now, evidence):
        """One correction: the deterministic reference, or a connectome proposal that passes the
        supervising envelope when the intent asked for that controller. Every step stays inside
        the same speed limit and client bounds; completion still needs measured arrival."""
        point = step(intent.cursor, goal, dt, intent.spec.speed_px_s, self.layout["rect"])
        intent.last_controller = "deterministic"
        if self.shadow is not None:
            from ganglion.brain.shadow import MotorSample
            stage = getattr(intent, "stage", "reach")
            if intent.spec.controller == "connectome":
                velocity = self.shadow.proposal((intent.id, stage, self.layout_rev), now)
                candidate = None if velocity is None else supervise(
                    intent.cursor, goal, velocity, dt, intent.spec.speed_px_s, self.layout["rect"],
                    intent.spec.tolerance_px)
                if candidate is not None:
                    point, intent.last_controller = candidate, "connectome"
                    intent.neural_commands += 1
                elif velocity is None:
                    intent.last_controller = "deterministic_stale"
                    intent.stale_commands += 1
                else:
                    intent.last_controller = "deterministic_override"
                    intent.overridden_commands += 1
            self.shadow.submit(MotorSample(intent.id, stage, self.layout_rev,
                evidence.observation_id, evidence.sample_started, now, min(self.expires, intent.expires),
                tuple(intent.cursor), tuple(goal), tuple(point), tuple(self.layout["rect"]),
                intent.spec.speed_px_s, dt, flow=self.lptc()))
        return point

    def lptc(self):
        """Wide-field flow for the model's lptc channel: only when enabled, a flow watch runs, its
        newest summary is credible (the view within the percept's range, the scene mostly
        agreeing with one motion) and fresh (a removed watch must not leave its last reading
        behind); otherwise None, which the adapter feeds as zeros."""
        if not self.lptc_feed or not self.flow or not self.flow.get("credible", True):
            return None
        if self.clock() - self.flow.get("captured_mono", self.clock()) > FLOW_MAX_AGE:
            return None
        f = self.flow
        return (f["tx_px_s"], f["ty_px_s"], f["divergence_s"], f["curl_s"])

    def snapshot(self):
        if self.frame is None:
            return None
        h, w = self.frame.shape[:2]
        return {"id": f"{self.ledger.epoch}:{self.layout_rev}:{self.frame_seq}", "change": self.change,
                "flow": self.flow,
                "observation_id": self.frame_seq, "captured_mono": self.frame_time,
                "capture_source": self.frame_source,
                "sample_started_mono": self.frame_started,
                "age_ms": max(0, (self.clock() - self.frame_time) * 1000),
                "session_id": self.session_id, "width": w, "height": h,
                "layout_revision": self.layout_rev, "target": self.layout,
                "coordinate_space": "screen", "origin": [0, 0]}

    def _check_snapshot(self, snapshot_id):
        if self.frame is None:
            self._error("no_frame", "Wait for the first captured frame and call look.")
        try:
            epoch, rev, seq = snapshot_id.split(":")
            valid = (epoch == self.ledger.epoch and int(rev) == self.layout_rev
                     and 0 < int(seq) <= self.frame_seq)
        except (ValueError, AttributeError):
            valid = False
        if not valid:
            self._error("stale_snapshot", "Call look and rebind against its snapshot ID.")

    def input(self, client, spec: InputSpec):
        """A discrete agent-chosen input, bound to a layout but not a visual predicate."""
        with self.lock:
            self._require_owner(client)
            self._check_snapshot(spec.snapshot_id)
            uses_pointer = spec.action in ("move", "click", "look", "button")
            if uses_pointer and (self.pending or (self.intent and self.intent.active)):
                self._error("pointer_busy", "Wait for pending output or cancel the active intent.")
            if spec.action in ("move", "click"):
                x, y = spec.point
                tx, ty, tw, th = self.layout["rect"]
                if not (tx <= x < tx + tw and ty <= y < ty + th
                        and x < self.frame.shape[1] and y < self.frame.shape[0]):
                    self._error("point_outside_target", "Input must be inside the bound client area.")
            cid = uuid4().hex
            now = self.clock()
            details = {"action": spec.action, "snapshot_id": spec.snapshot_id}
            command = {"command_id": cid, "target": self.layout, "deadline": min(self.expires, now + 0.05)}
            hold_until = min(self.expires, now + spec.hold_ms / 1000)
            keys = [spec.key] if spec.action == "key" else list(spec.keys)
            (self.key_pending if spec.action in ("key", "hold") else self.pending)[cid] = details
            try:
                if spec.action == "click":
                    self.output.submit(command | {"x": x, "y": y, "hold_ms": spec.hold_ms})
                elif spec.action == "move":
                    self.output.pointer(command | {"point": (x, y)})
                elif spec.action in ("key", "hold"):
                    self.output.key(command | {"keys": keys, "hold_until": hold_until})
                    if any(k in ("w", "a", "s", "d", "space", "ctrl", "lshift") for k in keys):
                        self.note_self_motion(hold_until + 0.35)   # the view settles after the keys go up
                elif spec.action == "look":
                    self.output.look(command | {"dx": spec.delta[0], "dy": spec.delta[1],
                                                "chunks": max(1, spec.spread_ms // 10)})
                    # Cover the capture pipeline: a turn shows up in captured frames a few frames late.
                    self.note_self_motion(now + spec.spread_ms / 1000 + 0.25)
                else:
                    self.output.button(command | {"button": spec.button, "hold_until": hold_until})
                    self.note_self_motion(hold_until + 0.15)   # recoil and muzzle flash are own motion
            except Exception as exc:
                self.pending.pop(cid, None)
                self.key_pending.pop(cid, None)
                self.halt(f"submit_failed: {exc}", degraded=True)
                self._error("input_unavailable", str(exc))
            self.ledger.append("input_requested", now, critical=True, command_id=cid,
                               action=spec.action, point=spec.point, keys=keys or None, delta=spec.delta,
                               button=spec.button if spec.action in ("click", "button") else None,
                               snapshot_id=spec.snapshot_id)
            return {"command_id": cid}

    def add_watch(self, client, spec: WatchSpec):
        with self.lock:
            self._require_owner(client)
            self._check_snapshot(spec.snapshot_id)
            x, y, w, h = spec.region
            tx, ty, tw, th = self.layout["rect"]
            if not (tx <= x and ty <= y and x + w <= tx + tw and y + h <= ty + th
                    and x + w <= self.frame.shape[1] and y + h <= self.frame.shape[0]):
                self._error("region_outside_target", "The watch must fit inside the target client area.")
            return {"watch_id": self._new_watch(spec)}

    def arm(self, client, spec: ArmSpec):
        with self.lock:
            self._require_owner(client)
            if spec.watch_id not in self.watches:
                self._error("unknown_watch", "Create a watch before arming its reflex.")
            if len(self.reflexes) >= 32:
                self._error("reflex_budget", "At most 32 reflexes; disarm one first.")
            rid = uuid4().hex
            self.reflexes[rid] = Reflex(spec, self.clock() + spec.ttl_seconds)
            self.state = "running"
            self.ledger.append("reflex_armed", self.clock(), critical=True, reflex_id=rid,
                               watch_id=spec.watch_id)
            return {"reflex_id": rid, "expires_mono": self.reflexes[rid].expires}

    def remove(self, client, ident, *, watch=False):
        with self.lock:
            self._require_owner(client)
            collection = self.watches if watch else self.reflexes
            if ident not in collection:
                self._error("unknown_id", "This watch/reflex no longer exists.")
            del collection[ident]
            if watch:
                self.reflexes = {k: v for k, v in self.reflexes.items() if v.spec.watch_id != ident}
                if self.intent and ident in (self.intent.spec.watch_id, self.intent.spec.destination_watch_id,
                                               self.intent.spec.until.watch_id if self.intent.spec.until else None):
                    self._finish_intent("failed", "watch_removed")
            self.ledger.append("watch_removed" if watch else "reflex_disarmed", self.clock(),
                               critical=True, id=ident)
            return {"removed": ident}

    def _new_watch(self, spec: WatchSpec):
        return new_watch(self, spec)

    def observe(self, frame, captured, seq, layout, *, source="provided", sample_started=None):
        """Apply one captured frame: detectors, ledger events and armed reflexes (see perception)."""
        observe(self, frame, captured, seq, layout, source=source, sample_started=sample_started)

    def status(self):
        with self.lock:
            self.tick()
            return {"runtime_id": self.ledger.epoch, "state": self.state, "reason": self.reason,
                    "lease": self.lease(), "snapshot": self.snapshot(),
                    "watches": [{"id": k, "name": v.spec.name, "present": v.present,
                                 "detection": v.detection} for k, v in self.watches.items()],
                    "reflexes": [{"id": k, "watch_id": v.spec.watch_id, "fires": v.fires,
                                  "expires_mono": v.expires} for k, v in self.reflexes.items()],
                    "pending_commands": list(self.pending), "pending_keys": list(self.key_pending),
                    "self_motion_until_mono": self.motion_until, "dropped_frames": self.dropped_frames,
                    "intent": self.intent.status() if self.intent else None,
                    "locomotion": self.locomotion.status() if self.locomotion else None,
                    "capture_sources": dict(self.source_counts),
                    "shadow": self.shadow.status() if self.shadow else None,
                    "latest_cursor": self.ledger.cursor()}
