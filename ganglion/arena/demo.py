"""Run the complete MCP -> core -> pixels -> reflex -> input -> Arena experiment."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from contextlib import ExitStack
from pathlib import Path

from ganglion.core.clock import percentiles
from ganglion.core.output import MemoryOutput, ProcessOutput
from ganglion.core.protocol import Server, Service
from ganglion.core.runner import Runner
from ganglion.core.runtime import Runtime
from .target import SyntheticArena, TARGET_RGB


def dependency_versions():
    versions = {}
    for name in ("mcp", "numpy", "opencv-python", "pygame", "dxcam", "playwright"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


async def exercise(endpoint, seconds):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters
    spec = StdioServerParameters(command=sys.executable, args=["-m", "ganglion.mcp", "--endpoint", str(endpoint)])
    async with Client(spec) as client:
        async def call(name, args=None):
            result = await client.call_tool(f"ganglion_{name}", args or {})
            if result.is_error:
                raise RuntimeError(str(result.content))
            return result.structured_content

        deadline = time.perf_counter() + 5
        snapshot = None
        while time.perf_counter() < deadline:
            status = await call("status")
            snapshot = status["snapshot"]
            if snapshot:
                break
            await asyncio.sleep(0.02)
        if snapshot is None:
            raise RuntimeError("no captured frame within five seconds")
        await call("claim", {"seconds": min(60, seconds + 5)})
        watched = await call("watch", {"spec": {
            "name": "green target", "snapshot_id": snapshot["id"],
            "region": snapshot["target"]["rect"], "color_rgb": list(TARGET_RGB),
            "tolerance": 15, "min_pixels": 100}})
        await call("arm", {"spec": {"watch_id": watched["watch_id"], "trigger": "present",
                                  "cooldown_ms": 250, "max_fires": 100,
                                  "ttl_seconds": min(60, seconds + 5)}})
        # The agent does no work during this interval. All reactions happen in the resident core.
        started = time.perf_counter()
        await asyncio.sleep(seconds)
        looked = await call("look", {"image": False, "limit": 200})
        events = looked["events"]
        lost = looked["lost_events"]
        while looked["has_more"]:
            looked = await call("look", {"image": False, "limit": 200, "cursor": looked["next_cursor"]})
            events.extend(looked["events"])
            lost += looked["lost_events"]
        before_halt = time.perf_counter()
        await call("halt")
        await asyncio.sleep(0.2)
        stopped = await call("status")
        return {"started_mono": started, "halt_requested_mono": before_halt,
                "events": events, "lost_events": lost, "final_state": stopped["state"],
                "dropped_frames": stopped["dropped_frames"]}


def summarize(result, truth):
    start, end = result["started_mono"], result["halt_requested_mono"]
    active = [e for e in truth if start <= e["t_mono"] <= end]
    submitted = [e for e in result["events"] if e["kind"] == "input_submitted"]
    stages = {"capture_to_submit_ms": [(e["t_mono"] - e["captured_mono"]) * 1000 for e in submitted],
              "percept_to_submit_ms": [(e["t_mono"] - e["percept_ready_mono"]) * 1000 for e in submitted]}
    scheduled = {e["target_id"]: e for e in truth if e["kind"] == "event_scheduled"}
    receives = [e for e in active if e["kind"] == "input_received"]
    stages["scheduled_to_receive_ms"] = [(e["t_mono"] - scheduled[e["target_id"]]["t_mono"]) * 1000
                                         for e in receives if e["target_id"] in scheduled
                                         and scheduled[e["target_id"]]["t_mono"] >= start]
    result["preexisting_targets_excluded_from_reaction_timing"] = sum(
        e["target_id"] in scheduled and scheduled[e["target_id"]]["t_mono"] < start for e in receives)
    result["score"] = {kind: sum(e["kind"] == kind for e in active)
                       for kind in ("hit", "miss", "false_action")}
    result["timing_ms"] = {key: percentiles(values) if values else {"n": 0} for key, values in stages.items()}
    result["truth"] = truth
    result["timing_notes"] = (
        "Monotonic host clock. render_submitted is measured after pygame flip, not physical presentation. "
        "Synthetic results exercise the pipeline but do not measure DXGI, SendInput, or OS scheduling. "
        "Effect-observed events mean target disappearance; Arena truth verifies actual task success.")
    result["passed"] = (result["score"]["hit"] >= 2 and result["score"]["false_action"] == 0
                        and result["score"]["miss"] == 0
                        and result["lost_events"] == 0 and result["final_state"] == "halted"
                        and not any(e["kind"] == "input_failed" for e in result["events"]))
    return result


def run(*, synthetic=False, seconds=6, path=None):
    if not 2 <= seconds <= 45:
        raise ValueError("demo duration must be 2–45 seconds")
    from ganglion import __version__
    with tempfile.TemporaryDirectory(prefix="ganglion-demo-") as temp, ExitStack() as stack:
        temp = Path(temp)
        arena_process = None
        if synthetic:
            capture = SyntheticArena()
            output = MemoryOutput(time.perf_counter, capture.click)
            target, session_id = capture.target, 0
        else:
            from ganglion.core.actuators import Actuators
            from ganglion.core.capture import Capture
            from ganglion.core.session import current
            from ganglion.core.window import api, describe
            info = current()
            session_id = info.session_id
            restore = Actuators()
            cursor, foreground = restore.cursor_pos(), api().GetForegroundWindow()

            def restore_desktop():
                restore.move_abs(*cursor)
                if foreground:
                    api().SetForegroundWindow(foreground)
            stack.callback(restore_desktop)
            stop, truth_path, ready = temp / "stop", temp / "truth.json", temp / "ready.json"
            log = stack.enter_context((temp / "arena.log").open("w"))
            arena_process = subprocess.Popen(
                [sys.executable, "-m", "ganglion.arena.target", "--ready", str(ready),
                 "--truth", str(truth_path), "--stop-file", str(stop),
                 "--seconds", str(seconds + 25)], stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

            def stop_arena():
                stop.touch()
                arena_process.wait(timeout=5)
            stack.callback(stop_arena)
            deadline = time.perf_counter() + 10
            while not ready.exists() and time.perf_counter() < deadline and arena_process.poll() is None:
                time.sleep(0.02)
            if not ready.exists():
                raise RuntimeError("Arena did not start: " + (temp / "arena.log").read_text())
            window = json.loads(ready.read_text())
            target = lambda: describe(window["hwnd"])
            capture, output = Capture(refresh_static=True), ProcessOutput()
        stack.callback(output.close)
        capture.start()
        stack.callback(capture.stop)
        runtime = Runtime(time.perf_counter, output, session_id=session_id)
        runner = Runner(runtime, capture, target).start()
        stack.callback(runner.close)
        service = Service(runtime)
        server = Server(service)
        stack.callback(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        stack.callback(server.shutdown)
        stack.callback(service.closed.set)
        endpoint = temp / "endpoint.json"
        endpoint.write_text(json.dumps(server.endpoint()))
        result = asyncio.run(exercise(endpoint, seconds))
        result.update({"version": __version__, "synthetic": synthetic, "session_id": session_id,
                       "seconds_without_agent_calls": seconds,
                       "measured_at": datetime.now(timezone.utc).isoformat(),
                       "python": sys.version.split()[0],
                       "dependencies": dependency_versions()})
        if synthetic:
            with capture.lock:
                truth = list(capture.world.events)
        else:
            # Stop evaluation before closing the Arena so target loss doesn't contaminate the result.
            runner.close()
            stop_arena()
            truth = json.loads(truth_path.read_text())
        result = summarize(result, truth)
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
