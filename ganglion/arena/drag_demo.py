"""MCP validation of static reach, drag-until, drop verification, rejection, and cancel."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from .demo import dependency_versions
from .harness import experiment
from .manipulation import ManipulationWorld, DESTINATION_RGB, CONDITION_RGB
from .target import TARGET_RGB


async def exercise(endpoint, reset, trials):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters
    transport = StdioServerParameters(command=sys.executable, args=["-m", "ganglion.mcp", "--endpoint", str(endpoint)])
    async with Client(transport) as client:
        async def call(name, args=None):
            result = await client.call_tool("ganglion_" + name, args or {})
            if result.is_error:
                raise RuntimeError(str(result.content))
            return result.structured_content

        cursor, events, lost = None, [], 0

        async def read():
            nonlocal cursor, lost
            while True:
                state = await call("look", {"cursor": cursor, "image": False, "limit": 200})
                cursor = state["next_cursor"]
                events.extend(state["events"])
                lost += state["lost_events"]
                if not state["has_more"]:
                    return state

        until = time.perf_counter() + 5
        while True:
            state = await read()
            if state["snapshot"]:
                break
            if time.perf_counter() >= until:
                raise RuntimeError("No captured frame")
            await asyncio.sleep(.02)
        await call("claim", {"seconds": 30})
        snapshot = state["snapshot"]
        watches = {}
        for name, color in (("source", TARGET_RGB), ("destination", DESTINATION_RGB), ("condition", CONDITION_RGB)):
            watches[name] = (await call("watch", {"spec": {"name": name, "snapshot_id": snapshot["id"],
                "region": snapshot["target"]["rect"], "color_rgb": list(color), "tolerance": 10,
                "min_pixels": 100}}))["watch_id"]
        records = []
        for seed in range(1, trials + 1):
            for mode in ("static_reach", "drag_until", "drop", "reject", "cancel"):
                await call("renew", {"seconds": 30})
                x, y, _, _ = snapshot["target"]["rect"]
                await call("input", {"spec": {"action": "move", "snapshot_id": snapshot["id"],
                                              "point": [x + 30, y + 100]}})
                await asyncio.sleep(.08)
                trial = {"id": f"{mode}-{seed}", "mode": mode, "seed": seed}
                await asyncio.to_thread(reset, trial)
                # No redraw heartbeat: this pause deliberately exhausts the old 250 ms bound.
                await asyncio.sleep(.5)
                state = await read()
                observed = {w["id"]: w for w in state["watches"]}
                destination = observed[watches["destination"]]["detection"]
                if not observed[watches["source"]]["present"] or destination is None:
                    raise RuntimeError("Fixture controls not observed on a quiet screen")
                before = cursor
                spec = {"program": "reach" if mode == "static_reach" else "drag",
                        "watch_id": watches["source"], "timeout_seconds": 5, "settle_ms": 80}
                if mode == "static_reach":
                    spec["click"] = True
                else:
                    spec.update(destination=[destination["x"], destination["y"]], speed_px_s=600,
                                until={"watch_id": watches["condition"]}, verification_seconds=.6)
                started = time.perf_counter()
                intent = await call("intent", {"spec": spec})
                if mode == "cancel":
                    await call("wait", {"cursor": before, "timeout": 3, "kinds": ["drag_started", "intent_failed"]})
                    await call("cancel", {"intent_id": intent["intent_id"]})
                    await call("wait", {"cursor": before, "timeout": 1, "kinds": ["output_halted"]})
                else:
                    await call("wait", {"cursor": before, "timeout": 5.5,
                        "kinds": ["intent_completed", "intent_failed", "intent_cancelled"]})
                state = await read()
                outcome = state["intent"]
                if outcome["phase"] in ("running", "clicking"):
                    await call("cancel", {"intent_id": intent["intent_id"]})
                elapsed = time.perf_counter() - started
                await asyncio.sleep(.15)
                state = await read()
                records.append(trial | {"elapsed_seconds": elapsed, "outcome": outcome,
                    "pending_commands": state["pending_commands"],
                    "button_held": state["intent"].get("button_held", False),
                    "capture_sources": state["capture_sources"]})
        await call("halt")
        await asyncio.sleep(.1)
        state = await read()
        return {"trials": records, "events": events, "lost_events": lost,
                "final_state": state["state"], "pending_commands": state["pending_commands"],
                "capture_sources": state["capture_sources"], "dropped_frames": state["dropped_frames"]}


def summarize(result, truth):
    for trial in result["trials"]:
        own = [e for e in truth if e.get("trial_id") == trial["id"]]
        trial["truth_counts"] = {kind: sum(e["kind"] == kind for e in own)
            for kind in ("pointer_down", "pointer_up", "hit", "drop_accepted", "drop_rejected", "false_action")}
        mode, outcome, counts = trial["mode"], trial["outcome"], trial["truth_counts"]
        if mode == "static_reach":
            passed = outcome["phase"] == "completed" and counts["hit"] == 1
        elif mode == "cancel":
            passed = outcome["phase"] == "cancelled" and counts["drop_rejected"] == 1
        elif mode == "reject":
            passed = outcome["phase"] == "failed" and outcome["reason"] == "condition_not_observed" and counts["drop_rejected"] == 1
        else:
            passed = outcome["phase"] == "completed" and outcome["condition_verified"] and counts["drop_accepted"] == 1
            if mode == "drag_until":
                drops = [e for e in own if e["kind"] == "drop_accepted"]
                passed = passed and bool(drops) and drops[0]["x"] < drops[0]["destination"][0] - 50
        trial["passed"] = (passed and counts["false_action"] == 0 and counts["pointer_down"] == 1
            and counts["pointer_up"] == 1 and not trial["pending_commands"] and not trial["button_held"])
    result["truth"] = truth
    result["score"] = {mode: {"passed": sum(t["passed"] for t in result["trials"] if t["mode"] == mode),
        "trials": sum(t["mode"] == mode for t in result["trials"])}
        for mode in ("static_reach", "drag_until", "drop", "reject", "cancel")}
    result["passed"] = (all(t["passed"] for t in result["trials"]) and not result["lost_events"]
        and result["final_state"] == "halted" and not result["pending_commands"]
        and not any(e["kind"] == "input_failed" for e in result["events"])
        and (result["environment"] == "synthetic" or result["capture_sources"].get("gdi_refresh", 0) > 0))
    result["notes"] = (
        "Five cases per seed through MCP: quiet-screen reach, early release on a visual condition, "
        "accepted drop, rejected drop with failed post-release verification, and cancellation while held. "
        "Fixture setup/truth are evaluator-only; policies see pixels and public tools. "
        "GDI fallback makes new acquisitions, with sample-start and availability times recorded separately. "
        "Synthetic results exclude real capture/input. This is fixture validation, not Solitaire gameplay.")
    return result


def run(*, environment="synthetic", trials=2, path=None):
    if not 1 <= trials <= 4:
        raise ValueError("Use 1–4 seeds")
    with experiment(environment, ManipulationWorld, scenario="manipulation") as host:
        result = asyncio.run(exercise(host.endpoint, host.reset, trials))
        result.update(environment=environment, session_id=host.session_id, browser_version=host.browser_version,
                      measured_at=datetime.now(timezone.utc).isoformat(), dependencies=dependency_versions(),
                      python=sys.version.split()[0])
        result = summarize(result, host.finish())
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
