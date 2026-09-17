"""Paired periodic-input / local-reach experiment through the real MCP stdio bridge."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from .demo import dependency_versions
from .reach_world import ReachWorld
from .target import TARGET_RGB


def neural_share(events):
    """How many pointer commands the connectome produced versus the supervisor overriding it."""
    feedback = [e for e in events if e["kind"] == "pointer_feedback" and "controller" in e]
    neural = sum(e["controller"] == "connectome" for e in feedback)
    overridden = sum(e["controller"] == "deterministic_override" for e in feedback)
    return {"pointer_commands": len(feedback), "connectome_commands": neural,
            "overridden_commands": overridden,
            "connectome_share": neural / (neural + overridden) if neural + overridden else None}


async def exercise(endpoint, reset, trials, controller="deterministic"):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters
    spec = StdioServerParameters(command=sys.executable, args=["-m", "ganglion.mcp", "--endpoint", str(endpoint)])
    async with Client(spec) as client:
        calls = 0

        async def call(name, args=None):
            nonlocal calls
            calls += 1
            result = await client.call_tool("ganglion_" + name, args or {})
            if result.is_error:
                raise RuntimeError(str(result.content))
            return result.structured_content

        cursor, events, lost = None, [], 0

        async def read():
            nonlocal cursor, lost
            while True:
                result = await call("look", {"cursor": cursor, "image": False, "limit": 200})
                cursor = result["next_cursor"]
                events.extend(result["events"])
                lost += result["lost_events"]
                if not result["has_more"]:
                    return result

        until = time.perf_counter() + 5
        while True:
            status = await read()
            if status["snapshot"]:
                break
            if time.perf_counter() >= until:
                raise RuntimeError("No captured frame")
            await asyncio.sleep(.02)
        await call("claim", {"seconds": 60})
        snapshot = status["snapshot"]
        wid = (await call("watch", {"spec": {"name": "moving choice", "snapshot_id": snapshot["id"],
            "region": snapshot["target"]["rect"], "color_rgb": list(TARGET_RGB),
            "min_pixels": 100, "tolerance": 15}}))["watch_id"]
        records = []
        for seed in range(1, trials + 1):
            for mode in ("periodic", "reach"):
                await call("renew", {"seconds": 30})
                x, y, _, _ = snapshot["target"]["rect"]
                await call("input", {"spec": {"action": "move", "snapshot_id": snapshot["id"],
                                              "point": [x + 30, y + 100]}})
                await asyncio.sleep(.06)
                trial = {"id": f"{mode}-{seed}", "seed": seed}
                await asyncio.to_thread(reset, trial)
                # Fixture setup is outside the policy: allow its reset to reach the display.
                await asyncio.sleep(.12)
                status = await read()
                watch = next(w for w in status["watches"] if w["id"] == wid)
                if not watch["present"]:
                    raise RuntimeError(f"Fixture target was not observed for {trial['id']}")
                first_call, started = calls, time.perf_counter()
                if mode == "reach":
                    intent = await call("intent", {"spec": {"watch_id": wid, "click": True,
                                                            "timeout_seconds": 3, "controller": controller}})
                    await call("wait", {"cursor": status["next_cursor"], "timeout": 3.5,
                        "kinds": ["intent_completed", "intent_failed", "intent_cancelled"]})
                    status = await read()
                    policy_calls = calls - first_call
                    outcome = status["intent"]
                    if outcome["phase"] in ("running", "clicking"):
                        await call("cancel", {"intent_id": intent["intent_id"]})
                else:
                    # Fixed budget: one point decision each 500 ms, with 250 ms deliberation.
                    # Watches/perception are identical to the local controller's.
                    attempts = 0
                    while attempts < 6 and watch["present"]:
                        point = [watch["detection"]["x"], watch["detection"]["y"]]
                        old_snapshot = status["snapshot"]["id"]
                        await asyncio.sleep(.25)
                        await call("input", {"spec": {"action": "click", "snapshot_id": old_snapshot,
                                                      "point": point}})
                        attempts += 1
                        await asyncio.sleep(.25)
                        status = await read()
                        watch = next(w for w in status["watches"] if w["id"] == wid)
                    policy_calls = calls - first_call
                    outcome = {"attempts": attempts, "target_absent": not watch["present"]}
                elapsed = time.perf_counter() - started
                await asyncio.sleep(.10)  # application event receipt and final output acknowledgement
                status = await read()
                records.append(trial | {"mode": mode, "policy_calls": policy_calls,
                    "elapsed_seconds": elapsed, "outcome": outcome})
        await call("halt")
        await asyncio.sleep(.1)
        status = await read()
        return {"trials": records, "events": events, "lost_events": lost,
                "final_state": status["state"], "pending_commands": status["pending_commands"],
                "dropped_frames": status["dropped_frames"]}


def summarize(result, truth):
    for trial in result["trials"]:
        own = [e for e in truth if e.get("trial_id") == trial["id"]]
        trial["hits"] = sum(e["kind"] == "hit" for e in own)
        trial["false_actions"] = sum(e["kind"] == "false_action" for e in own)
    scores = {}
    for mode in ("periodic", "reach"):
        trials = [t for t in result["trials"] if t["mode"] == mode]
        scores[mode] = {"trials": len(trials), "successes": sum(t["hits"] == 1 for t in trials),
            "false_actions": sum(t["false_actions"] for t in trials),
            "policy_calls": sum(t["policy_calls"] for t in trials),
            "mean_elapsed_seconds": sum(t["elapsed_seconds"] for t in trials) / len(trials)}
    result["score"] = scores
    result["truth"] = truth
    result["passed"] = (scores["reach"]["successes"] == scores["reach"]["trials"]
        and scores["reach"]["false_actions"] == 0 and result["lost_events"] == 0
        and result["final_state"] == "halted" and not result["pending_commands"]
        and all(t["outcome"]["phase"] == "completed" for t in result["trials"] if t["mode"] == "reach")
        and not any(e["kind"] == "input_failed" for e in result["events"]))
    result["notes"] = (
        "Paired trajectory seeds; six discrete attempts or one three-second reach per trial. "
        "Periodic baseline models 250 ms decision latency / 500 ms cadence, not measured LLM latency. "
        "Both policies use the public MCP interface and identical taught color perception. "
        "Fixture resets and ground truth are evaluator-only. Browser DOM is never a policy sensor. "
        "Elapsed time includes MCP overhead and completion observation; browser truth uses its own performance clock. "
        "Synthetic capture/output does not measure Windows latency. These are small fixture smoke tests, "
        "not evidence of general web navigation or a sub-tick tracking bound.")
    return result


def run(*, environment="synthetic", trials=4, path=None, shadow_checkpoint=None, controller="deterministic"):
    if not 1 <= trials <= 8:
        raise ValueError("Use 1–8 trials per policy")
    if controller == "connectome" and not shadow_checkpoint:
        raise ValueError("The connectome controller needs --shadow-checkpoint")
    from .harness import experiment
    predictor = None
    if shadow_checkpoint:
        from ganglion.brain.haltere_cursor import HaltereCursor
        predictor = HaltereCursor(shadow_checkpoint)
    with experiment(environment, ReachWorld, shadow_predictor=predictor) as host:
        result = asyncio.run(exercise(host.endpoint, host.reset, trials, controller))
        result.update({"environment": environment, "session_id": host.session_id,
            "measured_at": datetime.now(timezone.utc).isoformat(), "dependencies": dependency_versions(),
            "python": sys.version.split()[0],
            "controller": {"mode": controller, "gain_per_second": 35, "speed_px_s": 1200, "tolerance_px": 6,
                           "settle_ms": 30, "timeout_seconds": 3},
            "baseline": {"decision_delay_ms": 250, "minimum_cadence_ms": 500, "max_attempts": 6},
            "browser_version": host.browser_version})
        result = summarize(result, host.finish())
        result["shadow"] = host.shadow_status()
        if predictor:
            import numpy as np
            predictions = [e for e in result["events"] if e["kind"] == "shadow_prediction"]
            completed = [e for e in result["events"] if e["kind"] in ("shadow_prediction", "shadow_discarded")]
            result["shadow_score"] = {"predictions": len(predictions), "completed_inferences": len(completed),
                "promoted": False, "actuation_authority": "supervised_connectome" if controller == "connectome" else False,
                "neural_share": neural_share(result["events"]),
                "desktop_trained": predictor.metadata["desktop_trained"],
                "inference_ms": {f"p{p}": float(np.percentile([e["inference_ms"] for e in completed], p))
                                 for p in (50, 95, 99)} if completed else {},
                "within_5ms_fraction": sum(e["within_5ms"] for e in completed)/len(completed) if completed else None,
                "mean_disagreement_px": float(np.mean([e["disagreement_px"] for e in predictions])) if predictions else None}
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
