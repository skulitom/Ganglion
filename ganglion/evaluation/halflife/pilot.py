"""A persistent pilot for first-person play through the public MCP tools.

`serve` keeps one MCP session and lease alive, executes commands appended to commands.jsonl,
writes results to results.jsonl and saves frames. `do` appends one command and waits for its
result. Reflexes armed by one command stay armed while the agent thinks about the next one.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time



class PilotError(RuntimeError):
    pass


def align_max_step(c, *, default_turn_px_s=2500.0):
    """Mouse counts per tick for an align spec: an explicit `max_step`, else the counts that move
    the view `max_turn_px_s` in one 10 ms tick at the spec's gain."""
    if c.get("max_step") is not None:
        return int(c["max_step"])
    gain = float(c.get("gain", 1.2))
    return max(1, int(round(float(c.get("max_turn_px_s", default_turn_px_s)) * .01 * gain)))


class Pilot:
    def __init__(self, endpoint: str, directory: Path, *, lease_seconds: float = 45):
        self.endpoint, self.directory = endpoint, directory
        self.lease_seconds = lease_seconds
        self.session = None
        self.cursor = None
        self.snapshot = None
        self.origin = (0, 0)
        self.size = (0, 0)
        self.frames = 0
        self.watches: dict[str, str] = {}      # name -> watch_id
        self.reflexes: dict[str, str] = {}     # name -> reflex_id
        self.armed: dict[str, dict] = {}       # name -> arm spec, re-armed before its TTL ends
        self.armed_at: dict[str, float] = {}
        self.events_seen = 0
        self.log = (directory / "events.jsonl").open("a", encoding="utf-8")

    # -- transport ------------------------------------------------------------------------------
    async def _call(self, name, args=None, *, with_content=False):
        result = await self.session.call_tool("ganglion_" + name, args or {})
        if result.is_error:
            raise PilotError(f"{name}: " + " ".join(getattr(c, "text", "") for c in result.content))
        if with_content:
            return result.structured_content, result.content
        return result.structured_content

    def _record(self, events):
        for e in events:
            self.log.write(json.dumps(e) + "\n")
        self.events_seen += len(events)
        self.log.flush()

    async def drain(self):
        while True:
            more = await self._call("look", {"cursor": self.cursor, "image": False, "limit": 200})
            self.cursor = more["next_cursor"]
            self._record(more["events"])
            if not more["has_more"]:
                return more

    async def look(self, *, save=True, width=640):
        r, content = await self._call("look", {"cursor": self.cursor, "image": True, "limit": 200,
                                               "max_width": width, "image_format": "jpeg"}, with_content=True)
        self.cursor = r["next_cursor"]
        self._record(r["events"])
        self.snapshot = r["snapshot"]
        picture = next((c for c in content if getattr(c, "type", "") == "image"), None)
        path = None
        if picture is not None and save:
            self.frames += 1
            path = self.directory / f"frame-{self.frames:04d}.jpg"
            path.write_bytes(base64.b64decode(picture.data))
        return r, path

    async def status(self):
        return await self._call("status")

    async def wait(self, kinds, timeout):
        r = await self._call("wait", {"cursor": self.cursor, "timeout": timeout, "kinds": kinds})
        self.cursor = r["next_cursor"]
        self._record(r["events"])
        await self.drain()
        return r

    # -- commands ----------------------------------------------------------------------------------
    async def cmd_look(self, c):
        r, path = await self.look(width=c.get("width", 640))
        s = await self.status()
        return {"frame": str(path), "snapshot": r["snapshot"]["id"], "state": s["state"],
                "watches": {n: next((w["present"] for w in s["watches"] if w["id"] == i), None) for n, i in self.watches.items()},
                "reflexes": {n: next((x["fires"] for x in s["reflexes"] if x["id"] == i), None) for n, i in self.reflexes.items()},
                "intent": s["intent"] and {k: s["intent"].get(k) for k in ("program", "phase", "reason", "stage", "shots", "controller", "neural_commands", "overridden_commands")},
                "locomotion": s.get("locomotion") and {k: s["locomotion"].get(k) for k in ("keys", "phase", "reason", "renewals")},
                "pending": s["pending_commands"], "pending_keys": s["pending_keys"]}

    async def cmd_turn(self, c):
        await self._pointer_free()
        r = await self._call("input", {"spec": {"action": "look", "snapshot_id": self.snapshot["id"],
                                                "delta": [int(c.get("dx", 0)), int(c.get("dy", 0))],
                                                "spread_ms": int(c.get("spread_ms", 150))}})
        await self.wait(["look_done", "input_cancelled", "input_failed"], 1.0)
        return r

    async def cmd_aim(self, c):
        """Turn so that a point of the last full-resolution frame comes to the crosshair:
        {"x": px, "y": py, "counts_per_px": 1.067}. Yaw only unless "pitch" is true."""
        k = float(c.get("counts_per_px", 1.067))
        cx, cy = self.origin[0] + self.size[0] // 2, self.origin[1] + self.size[1] // 2
        dx = int(round((float(c["x"]) - cx) * k))
        dy = int(round((float(c.get("y", cy)) - cy) * k)) if c.get("pitch") else 0
        await self._pointer_free()
        r = await self._call("input", {"spec": {"action": "look", "snapshot_id": self.snapshot["id"],
                                                "delta": [max(-4000, min(4000, dx)), max(-4000, min(4000, dy))],
                                                "spread_ms": int(c.get("spread_ms", 250))}})
        await self.wait(["look_done", "input_cancelled", "input_failed"], 1.0)
        return {"delta": [dx, dy], "command_id": r["command_id"]}

    async def cmd_walk(self, c):
        keys = c.get("keys", ["w"])
        total = int(c.get("ms", 800))
        sent = 0
        started = time.perf_counter()
        while sent < total:
            chunk = min(900, total - sent)
            await self._call("input", {"spec": {"action": "hold", "snapshot_id": self.snapshot["id"],
                                                "keys": keys, "hold_ms": chunk}})
            sent += chunk
            # Re-issue before the hold expires so the walk is continuous but never unbounded.
            await asyncio.sleep(max(0.0, chunk / 1000 - 0.15))
        await asyncio.sleep(0.2)
        await self.drain()
        return {"keys": keys, "ms": total, "seconds": round(time.perf_counter() - started, 3)}

    async def cmd_tap(self, c):
        r = await self._call("input", {"spec": {"action": "key", "snapshot_id": self.snapshot["id"],
                                                "key": c["key"], "hold_ms": int(c.get("ms", 60))}})
        await asyncio.sleep(0.1)
        await self.drain()
        return r

    async def cmd_type(self, c):
        """Type text as bounded key taps (letters, digits, space, enter); for console commands."""
        names = {" ": "space", chr(10): "enter", "-": "minus", ".": "period", "_": "underscore"}
        typed = []
        for ch in c["text"]:
            key = names.get(ch, ch.lower())
            if key == "underscore":
                await self._call("input", {"spec": {"action": "hold", "snapshot_id": self.snapshot["id"],
                                                    "keys": ["lshift", "minus"], "hold_ms": 40}})
            else:
                await self._call("input", {"spec": {"action": "key", "snapshot_id": self.snapshot["id"],
                                                    "key": key, "hold_ms": int(c.get("ms", 30))}})
            typed.append(key)
            await asyncio.sleep(int(c.get("gap_ms", 45)) / 1000)
        if c.get("enter", True):
            await self._call("input", {"spec": {"action": "key", "snapshot_id": self.snapshot["id"], "key": "enter", "hold_ms": 30}})
        await asyncio.sleep(0.1)
        await self.drain()
        return {"typed": typed}

    async def cmd_fire(self, c):
        await self._pointer_free()
        r = await self._call("input", {"spec": {"action": "button", "snapshot_id": self.snapshot["id"],
                                                "button": c.get("button", "left"), "hold_ms": int(c.get("ms", 150))}})
        await self.wait(["input_released", "input_cancelled", "input_failed"], 1.5)
        return r

    async def cmd_watch(self, c):
        """Teach a watch: {"name", "kind": "color"|"motion", "region", ...spec fields}."""
        spec = {k: v for k, v in c.items() if k not in ("op", "name")}
        spec.setdefault("region", [self.origin[0], self.origin[1], self.size[0], self.size[1]])
        spec["snapshot_id"] = self.snapshot["id"]
        spec["name"] = c["name"]
        if c["name"] in self.watches:
            await self._call("unwatch", {"watch_id": self.watches[c["name"]]})
            self.reflexes = {n: r for n, r in self.reflexes.items() if not n.startswith(c["name"] + ":")}
        wid = (await self._call("watch", {"spec": spec}))["watch_id"]
        self.watches[c["name"]] = wid
        return {"watch_id": wid, "spec": spec}

    async def cmd_arm(self, c):
        """Arm a reflex on a named watch: {"watch", "response": "align"|"key"|"notify", ...}."""
        wid = self.watches[c["watch"]]
        spec = {k: v for k, v in c.items() if k not in ("op", "watch", "name")}
        spec["watch_id"] = wid
        name = c.get("name", f"{c['watch']}:{spec.get('response', 'click')}")
        if name in self.reflexes:
            try:
                await self._call("disarm", {"reflex_id": self.reflexes[name]})
            except PilotError:
                pass
        rid = (await self._call("arm", {"spec": spec}))["reflex_id"]
        self.reflexes[name] = rid
        self.armed[name], self.armed_at[name] = spec, time.perf_counter()
        return {"reflex_id": rid, "name": name}

    async def rearm(self):
        """Reflex TTLs are bounded to 60 s by the core; keep the ones the agent wants alive."""
        for name, spec in list(self.armed.items()):
            if time.perf_counter() - self.armed_at[name] > min(40, spec.get("ttl_seconds", 30) * 0.7):
                try:
                    await self._call("disarm", {"reflex_id": self.reflexes[name]})
                except PilotError:
                    pass
                self.reflexes[name] = (await self._call("arm", {"spec": spec}))["reflex_id"]
                self.armed_at[name] = time.perf_counter()

    async def cmd_engage(self, c):
        """Watch motion in a region and arm an align-and-fire reflex on it in one step.

        The align step is capped by `max_turn_px_s` (default 2500 px/s, where the flow percept
        still reports about 0.8 of the turn; it breaks near 7000) unless `max_step` is given.
        """
        region = c.get("region", [160, 60, 960, 440])
        watch = {"op": "watch", "name": c.get("name", "motion"), "kind": "motion", "region": region,
                 "min_pixels": int(c.get("min_pixels", 400)), "max_pixels": c.get("max_pixels", 200000),
                 "threshold": int(c.get("threshold", 30)), "lag_ms": int(c.get("lag_ms", 60))}
        w = await self.cmd_watch(watch)
        arm = {"op": "arm", "watch": watch["name"], "response": c.get("response", "track"), "trigger": c.get("trigger", "present"),
               "cooldown_ms": int(c.get("cooldown_ms", 400)), "max_fires": int(c.get("max_fires", 200)),
               "ttl_seconds": 60,
               "align": {"controller": c.get("controller", "deterministic"), "speed_px_s": float(c.get("speed_px_s", 1200)),
                         "tolerance_px": float(c.get("tolerance_px", 14)), "settle_ms": int(c.get("settle_ms", 20)),
                         "gain": float(c.get("gain", 1.2)), "max_step": align_max_step(c),
                         "absence_ms": int(c.get("absence_ms", 350)), "timeout_seconds": float(c.get("timeout_seconds", 4)),
                         "fire": {"button": "left", "hold_ms": int(c.get("fire_ms", 300)),
                                  "repeat": int(c.get("repeat", 4)), "interval_ms": int(c.get("interval_ms", 80))}}}
        a = await self.cmd_arm(arm)
        return {"watch": w, "reflex": a}

    async def cmd_disarm(self, c):
        names = [c["name"]] if c.get("name") else list(self.reflexes)
        out = {}
        for name in names:
            self.armed.pop(name, None)
            self.armed_at.pop(name, None)
            try:
                await self._call("disarm", {"reflex_id": self.reflexes.pop(name)})
                out[name] = "disarmed"
            except (PilotError, KeyError) as exc:
                out[name] = str(exc)
        return out

    async def cmd_align(self, c):
        """Start an align intent on a named watch and wait for it: {"watch", "fire": {...}, ...}."""
        await self._pointer_free()
        spec = {k: v for k, v in c.items() if k not in ("op", "watch")}
        spec.update(program="align", watch_id=self.watches[c["watch"]])
        spec.setdefault("timeout_seconds", 4)
        r = await self._call("intent", {"spec": spec})
        deadline = time.perf_counter() + spec["timeout_seconds"] + 1
        while time.perf_counter() < deadline:
            await self.wait(["intent_completed", "intent_failed", "intent_cancelled"], 0.5)
            s = await self.status()
            if s["intent"] and s["intent"]["phase"] not in ("running", "clicking"):
                return {k: s["intent"].get(k) for k in ("intent_id", "phase", "reason", "stage", "shots", "commands", "error_px", "applied_delta")}
        await self._call("cancel", {"intent_id": r["intent_id"]})
        return {"intent_id": r["intent_id"], "phase": "cancelled", "reason": "pilot_timeout"}

    async def cmd_move(self, c):
        """Start continuous locomotion the core renews itself: {"keys": [...], "s": seconds,
        "until": {"watch": name, "present": bool}}. Returns at once; the move runs on."""
        s = await self.status()
        if s.get("locomotion") and s["locomotion"]["phase"] == "running":
            await self._call("cancel", {"intent_id": s["locomotion"]["intent_id"]})
            await asyncio.sleep(0.05)
        spec = {"program": "move", "keys": c.get("keys", ["w"]), "timeout_seconds": float(c.get("s", 3))}
        if c.get("until"):
            spec["until"] = {"watch_id": self.watches[c["until"]["watch"]], "present": bool(c["until"].get("present", True))}
        r = await self._call("intent", {"spec": spec})
        if c.get("wait"):
            deadline = time.perf_counter() + spec["timeout_seconds"] + 1
            while time.perf_counter() < deadline:
                await self.wait(["intent_completed", "intent_failed", "intent_cancelled"], 0.5)
                s = await self.status()
                if s.get("locomotion") and s["locomotion"]["phase"] != "running":
                    return s["locomotion"]
        return r

    async def _watch_state(self, name):
        s = await self.status()
        wid = self.watches.get(name)
        for w in s["watches"]:
            if w["id"] == wid:
                return w, s
        return None, s

    async def cmd_explore(self, c):
        """Move on our own until a goal or motion watch is seen, bumping and turning when blocked.

        {"s": seconds, "goal": watch name or null, "motion": watch name or null, "bias": "left"|"right",
         "turn_deg": 60, "change_floor": 1.2}. Blocked means the upper view stops changing while the
        forward key is held; then the pilot turns by turn_deg, alternating sides with growing angle.
        """
        seconds = float(c.get("s", 20))
        goal, motion = c.get("goal"), c.get("motion", "motion" if "motion" in self.watches else None)
        sign = -1 if c.get("bias", "left") == "left" else 1
        turn_deg = float(c.get("turn_deg", 60))
        floor = float(c.get("change_floor", 1.2))
        counts_per_deg = float(c.get("counts_per_deg", 15.15))
        started = time.perf_counter()
        bumps, moves = 0, 0
        outcome = "timeout"
        while time.perf_counter() - started < seconds:
            remaining = seconds - (time.perf_counter() - started)
            await self._call("intent", {"spec": {"program": "move", "keys": ["w"], "timeout_seconds": max(0.5, min(3.0, remaining))}})
            moves += 1
            move_started = time.perf_counter()
            quiet = 0
            stop_reason = None
            while time.perf_counter() - move_started < 3.0 and time.perf_counter() - started < seconds:
                await asyncio.sleep(0.1)
                s = await self.status()
                present = {w["id"]: w for w in s["watches"]}
                if goal and self.watches.get(goal) in present and present[self.watches[goal]]["present"]:
                    stop_reason = "goal"
                    break
                if motion and self.watches.get(motion) in present and present[self.watches[motion]]["present"]:
                    stop_reason = "motion"
                    break
                loco = s.get("locomotion")
                if not loco or loco["phase"] != "running":
                    break
                change = (s.get("snapshot") or {}).get("change")
                if time.perf_counter() - move_started > 0.5 and change is not None and change < floor:
                    quiet += 1
                    if quiet >= 5:
                        stop_reason = "blocked"
                        break
                else:
                    quiet = 0
            s = await self.status()
            if s.get("locomotion") and s["locomotion"]["phase"] == "running":
                await self._call("cancel", {"intent_id": s["locomotion"]["intent_id"]})
            await self.drain()
            if stop_reason in ("goal", "motion"):
                outcome = stop_reason
                break
            if stop_reason == "blocked":
                bumps += 1
                import random
                angle = random.uniform(0.6, 1.6) * turn_deg * (sign if random.random() < 0.7 else -sign)
                if bumps % 4 == 0:
                    angle = 180 * (1 if random.random() < 0.5 else -1)
                await self._call("input", {"spec": {"action": "look", "snapshot_id": self.snapshot["id"],
                                                    "delta": [int(angle * counts_per_deg), 0], "spread_ms": 250}})
                await self.wait(["look_done", "input_cancelled", "input_failed"], 1.0)
                await asyncio.sleep(0.3)
        return {"outcome": outcome, "bumps": bumps, "moves": moves, "seconds": round(time.perf_counter() - started, 2)}

    async def cmd_seek(self, c):
        """Explore until the goal watch is seen, align on it, then approach until it looms."""
        goal = c["goal"]
        near = int(c.get("near_pixels", 20000))
        result = await self.cmd_explore(dict(c, motion=c.get("motion")))
        if result["outcome"] != "goal":
            return result
        align = await self.cmd_align({"op": "align", "watch": goal, "tolerance_px": float(c.get("tolerance_px", 40)),
                                      "settle_ms": 20, "gain": 1.1, "max_step": 250, "absence_ms": 600, "timeout_seconds": 4})
        approach_started = time.perf_counter()
        while time.perf_counter() - approach_started < float(c.get("approach_s", 8)):
            w, s = await self._watch_state(goal)
            if not w or not w["present"]:
                break
            if w["detection"]["pixels"] >= near:
                await self.cmd_stop({})
                return {"outcome": "reached", "align": align, "pixels": w["detection"]["pixels"]}
            if not s.get("locomotion") or s["locomotion"]["phase"] != "running":
                await self._call("intent", {"spec": {"program": "move", "keys": ["w"], "timeout_seconds": 2}})
            await asyncio.sleep(0.15)
        await self.cmd_stop({})
        return {"outcome": "lost_or_timeout", "align": align}

    async def cmd_calibrate(self, c):
        """Sweep the view at known rates and sample the flow watch: the percept's scale, lag and
        valid range against the turn the runtime applied.

        {"rates": [300, 600, 1000, 1500, 2500], "seconds": 0.6, "counts_per_px": 1.067, "pause": 0.4}
        Each rate is swept right then left as one spread `look`; the snapshot's flow summary is
        sampled about every 20 ms during the sweep and for a while after it. Writes
        calibration.json beside the events and returns a per-sweep summary.
        """
        import numpy as np
        await self._pointer_free()
        rates = [float(r) for r in c.get("rates", [300, 600, 1000, 1500, 2500])]
        seconds = float(c.get("seconds", 0.6))
        cpp = float(c.get("counts_per_px", 1.067))
        pause = float(c.get("pause", 0.4))
        samples, sweeps = [], []

        async def sample_until(until, sweep, expected):
            while time.perf_counter() < until:
                s = await self.status()
                snap = s.get("snapshot") or {}
                samples.append({"t": time.perf_counter(), "captured": snap.get("captured_mono"), "sweep": sweep,
                                "expected_px_s": expected, "flow": snap.get("flow")})
                await asyncio.sleep(0.015)

        for rate in rates:
            for sign in (1, -1):
                counts = int(round(sign * rate * cpp * seconds))
                expected = -sign * rate                     # the picture moves against the view
                index = len(sweeps)
                sweep = {"rate_px_s": rate, "direction": sign, "counts": counts, "spread_ms": int(seconds * 1000)}
                sweeps.append(sweep)
                started = time.perf_counter()
                await self._call("input", {"spec": {"action": "look", "snapshot_id": self.snapshot["id"],
                                                    "delta": [counts, 0], "spread_ms": int(seconds * 1000)}})
                sweep["started"] = started
                await sample_until(started + seconds + 0.2, index, expected)
                await self.wait(["look_done", "input_cancelled", "input_failed"], 1.0)
                sweep["ended"] = time.perf_counter()
                await sample_until(time.perf_counter() + pause, index, 0.0)
                await self.drain()
        for i, sweep in enumerate(sweeps):
            steady = [x for x in samples if x["sweep"] == i and x["flow"]
                      and sweep["started"] + 0.15 <= x["t"] <= sweep["started"] + seconds]
            tx = [x["flow"]["tx_px_s"] for x in steady]
            expected = -sweep["direction"] * sweep["rate_px_s"]
            sweep["samples"] = len(steady)
            sweep["flow_tx_median"] = float(np.median(tx)) if tx else None
            sweep["ratio"] = float(np.median(tx) / expected) if tx else None
            sweep["inlier_median"] = float(np.median([x["flow"]["inlier_fraction"] for x in steady])) if steady else None
            sweep["response_median"] = float(np.median([x["flow"].get("phase_response", 0) for x in steady])) if steady else None
            onset = next((x["t"] - sweep["started"] for x in samples
                          if x["sweep"] == i and x["flow"] and abs(x["flow"]["tx_px_s"]) > 0.5 * sweep["rate_px_s"]), None)
            sweep["onset_s"] = onset
        report = {"counts_per_px": cpp, "sweeps": sweeps, "samples": samples}
        path = self.directory / "calibration.json"
        path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        return {"path": str(path), "sweeps": [{k: v for k, v in sw.items() if k not in ("started", "ended")} for sw in sweeps]}

    async def cmd_stop(self, c):
        s = await self.status()
        if s.get("locomotion") and s["locomotion"]["phase"] == "running":
            await self._call("cancel", {"intent_id": s["locomotion"]["intent_id"]})
            await self.drain()
            return {"stopped": s["locomotion"]["keys"]}
        return {"stopped": []}

    async def cmd_reflexes(self, c):
        """Arm the bounded key reflexes for a Half-Life style HUD: reload and weapon switch when
        the clip counter turns red, and a strafe when the damage indicator flashes by the crosshair.
        Regions are client pixels of a 1280x720 view; pass overrides to adapt."""
        x, y = self.origin
        out = {}
        clip = c.get("clip_region", [1090, 676, 110, 40])
        w = await self.cmd_watch({"op": "watch", "name": "clip_red", "kind": "color",
                                  "region": [x + clip[0], y + clip[1], clip[2], clip[3]],
                                  "color_rgb": c.get("red_rgb", [255, 20, 20]), "tolerance": int(c.get("red_tolerance", 70)),
                                  "min_pixels": int(c.get("clip_min_pixels", 60))})
        out["clip_red"] = w["watch_id"]
        out["reload"] = await self.cmd_arm({"op": "arm", "watch": "clip_red", "name": "reload", "response": "key",
                                            "key": "r", "hold_ms": 60, "trigger": "present", "cooldown_ms": int(c.get("reload_cooldown_ms", 2500)),
                                            "max_fires": 200, "ttl_seconds": 60})
        out["switch"] = await self.cmd_arm({"op": "arm", "watch": "clip_red", "name": "switch", "response": "key",
                                            "key": c.get("switch_key", "2"), "hold_ms": 60, "trigger": "present",
                                            "cooldown_ms": int(c.get("switch_cooldown_ms", 6000)), "max_fires": 50, "ttl_seconds": 60})
        hit = c.get("hit_region", [520, 240, 240, 240])
        w = await self.cmd_watch({"op": "watch", "name": "hit", "kind": "color",
                                  "region": [x + hit[0], y + hit[1], hit[2], hit[3]],
                                  "color_rgb": c.get("hit_rgb", [230, 30, 30]), "tolerance": int(c.get("hit_tolerance", 60)),
                                  "min_pixels": int(c.get("hit_min_pixels", 120))})
        out["hit"] = w["watch_id"]
        out["dodge"] = await self.cmd_arm({"op": "arm", "watch": "hit", "name": "dodge", "response": "key",
                                           "key": c.get("dodge_key", "a"), "hold_ms": int(c.get("dodge_ms", 350)),
                                           "trigger": "appear", "cooldown_ms": int(c.get("dodge_cooldown_ms", 800)),
                                           "max_fires": 200, "ttl_seconds": 60})
        return out

    async def cmd_cancel(self, c):
        s = await self.status()
        if s["intent"] and s["intent"]["phase"] in ("running", "clicking"):
            await self._call("cancel", {"intent_id": s["intent"]["intent_id"]})
            await self.wait(["output_halted"], 1)
        return {"cancelled": bool(s["intent"])}

    async def cmd_wait(self, c):
        """Observe for a while: report reflex firings and intents that happened meanwhile."""
        seconds = float(c.get("s", 2))
        started = time.perf_counter()
        fired, intents = 0, []
        while time.perf_counter() - started < seconds:
            r = await self.wait(["reflex_fired", "intent_completed", "intent_failed", "notify"],
                                min(1.0, seconds - (time.perf_counter() - started)))
            for e in r["events"]:
                if e["kind"] == "reflex_fired":
                    fired += 1
                elif e["kind"] in ("intent_completed", "intent_failed"):
                    intents.append({k: e.get(k) for k in ("kind", "reason", "shots", "program")})
        return {"reflex_fired": fired, "intents": intents, "seconds": round(time.perf_counter() - started, 2)}

    async def _pointer_free(self, timeout=2.0):
        until = time.perf_counter() + timeout
        while time.perf_counter() < until:
            s = await self.status()
            if not s["pending_commands"] and not (s["intent"] and s["intent"]["phase"] in ("running", "clicking")):
                return
            await asyncio.sleep(0.03)
        raise PilotError("pointer is busy (an intent or pending output)")

    async def execute(self, c):
        handler = getattr(self, "cmd_" + c["op"], None)
        if handler is None:
            raise PilotError(f"unknown op {c['op']}")
        if self.snapshot is None or c["op"] != "look":
            await self.look(save=False)
        return await handler(c)

    # -- serving ---------------------------------------------------------------------------------------
    async def serve(self, seconds: float, expected_session: int | None):
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        params = StdioServerParameters(command=sys.executable, args=["-m", "ganglion.mcp", "--endpoint", str(self.endpoint)])
        commands = self.directory / "commands.jsonl"
        results = self.directory / "results.jsonl"
        commands.touch()
        done = 0
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
                self.cursor = (await self.status())["latest_cursor"]
                r, _ = await self.look(save=False)
                snap = self.snapshot
                if expected_session is not None and snap["session_id"] != expected_session:
                    raise PilotError(f"core runs in session {snap['session_id']}, expected {expected_session}")
                x, y, w, h = snap["target"]["rect"]
                self.origin, self.size = (x, y), (w, h)
                await self._call("claim", {"seconds": self.lease_seconds})
                last_renew = time.perf_counter()
                keepalive = asyncio.create_task(self._keepalive())
                deadline = time.perf_counter() + seconds
                (self.directory / "ready.json").write_text(json.dumps({"target": snap["target"], "session": snap["session_id"]}))
                print(f"pilot ready: {w}x{h} client at {x},{y} in session {snap['session_id']}", flush=True)
                try:
                    while time.perf_counter() < deadline:
                        if time.perf_counter() - last_renew > self.lease_seconds / 2:
                            await self._call("renew", {"seconds": self.lease_seconds})
                            last_renew = time.perf_counter()
                        lines = commands.read_text(encoding="utf-8").splitlines()
                        if len(lines) > done:
                            c = json.loads(lines[done])
                            done += 1
                            started = time.perf_counter()
                            try:
                                result = await self.execute(c)
                                out = {"ok": True, "result": result}
                            except Exception as exc:
                                out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                            out.update(index=done - 1, op=c["op"], seconds=round(time.perf_counter() - started, 3),
                                       at=datetime.now(timezone.utc).isoformat())
                            with results.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(out) + "\n")
                            print(json.dumps(out)[:300], flush=True)
                            if c["op"] == "halt":
                                break
                        else:
                            await self.drain()
                            await self.rearm()
                            await asyncio.sleep(0.05)
                finally:
                    keepalive.cancel()
                    try:
                        await self._call("halt")
                    except Exception:
                        pass
                    self.log.close()

    async def cmd_halt(self, c):
        return {"halting": True}

    async def _keepalive(self):
        """Renew the lease on its own schedule so a long command cannot let it lapse."""
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            try:
                await self._call("renew", {"seconds": self.lease_seconds})
            except Exception as exc:
                print(f"keepalive: {exc}", flush=True)


def do(directory: Path, command: dict, timeout: float) -> dict:
    commands = directory / "commands.jsonl"
    results = directory / "results.jsonl"
    index = len(commands.read_text(encoding="utf-8").splitlines()) if commands.exists() else 0
    with commands.open("a", encoding="utf-8") as f:
        f.write(json.dumps(command) + "\n")
    until = time.perf_counter() + timeout
    while time.perf_counter() < until:
        if results.exists():
            for line in results.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("index") == index:
                    return r
        time.sleep(0.05)
    return {"ok": False, "error": "timeout waiting for the pilot", "index": index}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--endpoint", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--seconds", type=float, default=3600)
    s.add_argument("--session", type=int)
    d = sub.add_parser("do")
    d.add_argument("--dir", required=True)
    d.add_argument("command", help="JSON object with an op")
    d.add_argument("--timeout", type=float, default=30)
    a = parser.parse_args()
    directory = Path(a.dir)
    directory.mkdir(parents=True, exist_ok=True)
    if a.cmd == "serve":
        asyncio.run(Pilot(a.endpoint, directory).serve(a.seconds, a.session))
        return 0
    result = do(directory, json.loads(a.command), a.timeout)
    print(json.dumps(result, indent=1))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
