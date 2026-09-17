"""Run an explicitly taught board profile through the public MCP interface.

Attaches to a separately launched core. Never launches, focuses, deals, resets, or closes an app.
Profiles describe a particular observed layout, not general recognition or a game solver.
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

from pydantic import Field, model_validator

from ganglion.core.schema import Model, WatchSpec, IntentSpec
from ganglion.arena.demo import dependency_versions


class Teaching(Model):
    region: tuple[int, int, int, int]
    color_rgb: tuple[int, int, int]
    tolerance: int = 15
    min_pixels: int = 50


class Case(Model):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    description: str
    source: str
    destination: tuple[int, int]
    condition: str
    condition_present: bool = True
    expected_phase: str = "completed"
    expected_reason: str = "condition_observed_after_release"
    before: dict[str, bool] = Field(min_length=1)
    after: dict[str, bool] = Field(min_length=1)


class Profile(Model):
    name: str
    application: str
    expected_session: int
    client_size: tuple[int, int]
    watches: dict[str, Teaching]
    cases: list[Case] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def references(self):
        for name, taught in self.watches.items():
            WatchSpec(name=name, snapshot_id="validation", **taught.model_dump())
            x, y, w, h = taught.region
            if x + w > self.client_size[0] or y + h > self.client_size[1]:
                raise ValueError("teaching is outside the expected client")
        for case in self.cases:
            refs = {case.source, case.condition, *case.before, *case.after}
            if not refs <= self.watches.keys():
                raise ValueError("case references an untaught watch")
            if case.source not in case.before or not case.before[case.source]:
                raise ValueError("each case must require its source before input")
            if case.before.get(case.condition) != (not case.condition_present):
                raise ValueError("require an unsatisfied condition before input")
            x, y = case.destination
            if not (0 <= x < self.client_size[0] and 0 <= y < self.client_size[1]):
                raise ValueError("destination is outside the expected client")
            IntentSpec(program="drag", watch_id=case.source, destination=case.destination,
                       until={"watch_id": case.condition, "present": case.condition_present})
        if len({c.name for c in self.cases}) != len(self.cases):
            raise ValueError("case names must be unique")
        return self


def predicate_checks(state, ids, expected):
    watches = {w["id"]: w for w in state["watches"]}
    return {name: {"expected": value, "observed": watches.get(ids[name], {}).get("present"),
                   "passed": watches.get(ids[name], {}).get("present") == value}
            for name, value in expected.items()}


async def exercise(endpoint, profile, directory):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    report = {"profile": profile.model_dump(), "measured_at": datetime.now(timezone.utc).isoformat(),
              "dependencies": dependency_versions(), "python": sys.version.split()[0],
              "cases": [], "events": [], "lost_events": 0, "passed": False,
              "verification": "Taught pixel predicates and saved images; no application-internal truth or solver."}
    transport = StdioServerParameters(command=sys.executable,
        args=["-m", "ganglion.mcp", "--endpoint", str(endpoint)])
    async with Client(transport) as client:
        async def call(name, args=None):
            result = await client.call_tool("ganglion_" + name, args or {})
            if result.is_error:
                raise RuntimeError(str(result.content))
            return result

        cursor = None

        async def read(picture=None):
            nonlocal cursor
            while True:
                result = await call("look", {"cursor": cursor, "image": bool(picture), "limit": 200})
                state = result.structured_content
                report["events"].extend(state["events"])
                report["lost_events"] += state["lost_events"]
                cursor = state["next_cursor"]
                if picture:
                    for content in result.content:
                        if content.type == "image":
                            (directory / (picture + ".jpg")).write_bytes(base64.b64decode(content.data))
                    picture = None
                if not state["has_more"]:
                    return state

        claimed = False
        try:
            state = await read("initial")
            snapshot = state["snapshot"]
            if not snapshot:
                raise ValueError("core has no captured frame")
            if snapshot["session_id"] != profile.expected_session:
                raise ValueError("core is in a different Windows session")
            x, y, w, h = snapshot["target"]["rect"]
            if (w, h) != profile.client_size:
                raise ValueError("client size differs from the taught profile")
            report["target"] = snapshot["target"]
            report["session_id"] = snapshot["session_id"]
            report["capture_sources_at_start"] = state["capture_sources"]
            report["dropped_frames_at_start"] = state["dropped_frames"]
            await call("claim", {"seconds": 30})
            claimed = True
            ids = {}
            for name, taught in profile.watches.items():
                tx, ty, tw, th = taught.region
                spec = taught.model_dump() | {"name": name, "snapshot_id": snapshot["id"],
                                             "region": [x + tx, y + ty, tw, th]}
                ids[name] = (await call("watch", {"spec": spec})).structured_content["watch_id"]
            report["watch_ids"] = ids
            await asyncio.sleep(.3)
            for case in profile.cases:
                await call("renew", {"seconds": 30})
                state = await read(case.name + "-before")
                checks = predicate_checks(state, ids, case.before)
                record = {"name": case.name, "description": case.description, "before": checks, "passed": False}
                report["cases"].append(record)
                if not all(c["passed"] for c in checks.values()):
                    record["error"] = "precondition_failed; no input submitted"
                    break
                spec = {"program": "drag", "watch_id": ids[case.source],
                    "destination": [x + case.destination[0], y + case.destination[1]],
                    "until": {"watch_id": ids[case.condition], "present": case.condition_present},
                    "timeout_seconds": 8, "verification_seconds": 1.2,
                    "settle_ms": 100, "speed_px_s": 500, "tolerance_px": 5}
                started = time.perf_counter()
                record["intent"] = (await call("intent", {"spec": spec})).structured_content
                while time.perf_counter() - started < 9:
                    await call("wait", {"cursor": cursor, "timeout": .2,
                        "kinds": ["intent_completed", "intent_failed", "intent_cancelled"]})
                    state = await read()
                    if state["intent"]["phase"] not in ("running", "clicking"):
                        break
                if state["intent"]["phase"] in ("running", "clicking"):
                    await call("cancel", {"intent_id": record["intent"]["intent_id"]})
                record["elapsed_seconds"] = time.perf_counter() - started
                # A separate delayed observation checks source/destination after the intent ended.
                await asyncio.sleep(.4)
                state = await read(case.name + "-after")
                record.update(outcome=state["intent"], after=predicate_checks(state, ids, case.after),
                              snapshot=state["snapshot"], pending_commands=state["pending_commands"])
                outcome = record["outcome"]
                ident = outcome["intent_id"]
                own = [e for e in report["events"] if e.get("drag_id") == ident]
                record["drag_starts"] = sum(e["kind"] == "drag_started" for e in own)
                record["releases"] = sum(e["kind"] == "input_released" for e in own)
                record["post_observation_after_release"] = (
                    state["snapshot"]["sample_started_mono"] > outcome["released_mono"])
                record["passed"] = (outcome["phase"] == case.expected_phase
                    and outcome["reason"] == case.expected_reason
                    and all(c["passed"] for c in record["after"].values())
                    and not outcome["button_held"] and not state["pending_commands"]
                    and record["drag_starts"] == record["releases"] == 1
                    and record["post_observation_after_release"])
                if not record["passed"]:
                    break
            report["passed"] = (len(report["cases"]) == len(profile.cases)
                and all(c["passed"] for c in report["cases"]) and not report["lost_events"]
                and not any(e["kind"] == "input_failed" for e in report["events"]))
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if claimed:
                try:
                    await call("halt")
                    await asyncio.sleep(.15)
                    state = await read()
                    report.update(final_state=state["state"], pending_commands=state["pending_commands"],
                        capture_sources=state["capture_sources"], dropped_frames=state["dropped_frames"])
                    report["passed"] &= (state["state"] == "halted" and not state["pending_commands"]
                        and not report["lost_events"]
                        and not any(e["kind"] == "input_failed" for e in report["events"]))
                except Exception as exc:
                    report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                    report["passed"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True, help="New output directory; existing results are never overwritten")
    args = parser.parse_args()
    profile = Profile.model_validate_json(Path(args.profile).read_text(encoding="utf-8"))
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    result = asyncio.run(exercise(args.endpoint, profile, directory))
    (directory / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("events", "profile")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
