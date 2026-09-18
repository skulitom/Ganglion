"""Encounters at range from staged saves, run in randomised pairs, scored from the save file.

The point-blank trials (`trials.py`) cannot separate two controllers: the enemy is spawned on
the player, the turn is over in a second, and the game's autoaim (on below the hard skill)
steers every shot within five degrees onto the body. A scenario here is a save staged once at a
real encounter: hard skill (no autoaim, grunts at 80 health), god mode and notarget on, the
submachine gun in hand, the enemies in view and unaware. A trial touches no console:

    tap the scenario's load key, settle, arm the engage chain, tap the stimulus key (notarget off:
    the enemies see the player within a tenth of a second), watch for the window, disarm, tap the
    key bound to `save gltrial`, and read that save: the fall in the enemies' total health is the
    damage dealt, the dead are the kills, and the player's flags prove the trial was the staged one.

Trials run in pairs, one per condition in a seeded random order, under one core, so a session's
drift falls on both conditions alike; the analysis takes the difference inside each pair, the
median of those per session, and a sign-flip test across sessions, because one session is one
sample (docs/bench/HALFLIFE.md, "Five sessions"). A scenario whose engage settings are not yet
frozen from reference-only trials runs the reference alone.

    python -m ganglion.evaluation.halflife.encounters install --game <game dir>
    python -m ganglion.evaluation.halflife.encounters check --game <game dir> --scenario ledge-trio
    python -m ganglion.evaluation.halflife.encounters run --dir runs/hl/pilot --game <game dir> --scenario ledge-trio \\
        --label ledge-aa-1 --pairs 15 --seed 1 --conditions ref-a=deterministic,ref-b=deterministic
    python -m ganglion.evaluation.halflife.encounters analyse runs/hl/encounters/ledge-v6-*.json

Launch the game with `+sv_cheats 1 +exec ganglion.cfg` so the binds exist and the stimulus works.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import random
import re
import shutil
import time

import numpy as np

from . import saves
from .trials import slice_summary, step

SCENARIOS = Path(__file__).with_name("scenarios.json")
TRIAL_SAVE, SAVE_KEY, CHEATS_KEY = "gltrial", "f4", "f8"
FORBIDDEN_KEYS = {"f5", "f6", "f10", "f12"}        # snapshot, the user's quicksave, quit, the Steam overlay's screenshot
ENGAGE_KEYS = {"region", "min_pixels", "max_pixels", "threshold", "lag_ms", "tolerance_px", "settle_ms", "absence_ms",
               "timeout_seconds", "fire_ms", "repeat", "interval_ms", "cooldown_ms", "gain"}
CFG_HEAD = '''// Written by ganglion.evaluation.halflife.encounters install. Launch with: +sv_cheats 1 +exec ganglion.cfg
alias level "cl_pitchup 0;cl_pitchdown 0;wait;wait;wait;wait;wait;cl_pitchup 89;cl_pitchdown 89"
bind end level
hud_fastswitch 1
m_rawinput 1
m_filter 0
joystick 0
bind f8 "sv_cheats 1"
bind f9 "give monster_human_grunt"
bind f4 "save gltrial"
bind f11 "notarget"
// staging only: letters would be typed into an open console, so no trial taps them
bind g "god"
bind i "impulse 101"
bind m "weapon_9mmAR"
'''


def load_scenarios(path=SCENARIOS) -> dict:
    scenarios = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate(scenarios)
    if problems:
        raise ValueError("; ".join(problems))
    return scenarios


def validate(scenarios) -> list[str]:
    """What is wrong with a scenario file; [] when nothing. Trial-time keys are function keys (a
    letter would be typed into a console left open), none of them the game's or the user's own."""
    problems, keys = [], {SAVE_KEY: "the trial save", CHEATS_KEY: "sv_cheats"}
    for name, s in scenarios.items():
        for field in ("map", "save", "load_key", "enemy", "enemies", "settle_s", "window_s", "primary", "engage", "frozen"):
            if field not in s:
                problems.append(f"{name}: no {field}")
        if not re.fullmatch(r"[a-z][a-z0-9]{2,15}", str(s.get("save", ""))) or s.get("save") == TRIAL_SAVE:
            problems.append(f"{name}: save name {s.get('save')!r} must be lowercase letters and digits, and not the trial save")
        for field in ("load_key", "stimulus_key"):
            key = s.get(field)
            if key is None and field == "stimulus_key":
                continue
            if not re.fullmatch(r"f([1-9]|1[0-2])", str(key)) or key in FORBIDDEN_KEYS:
                problems.append(f"{name}: {field} {key!r} must be a function key other than {sorted(FORBIDDEN_KEYS)}")
            elif field == "load_key" and keys.setdefault(key, name) != name:
                problems.append(f"{name}: {key} already loads {keys[key]}")
        if s.get("stimulus_key") not in (None, "f11"):
            problems.append(f"{name}: the only stimulus bound is f11 (notarget)")
        if s.get("primary") not in ("damage", "first_shot_s"):
            problems.append(f"{name}: primary must be damage or first_shot_s")
        if not 1 <= float(s.get("window_s", 0)) <= 60 or not 0 <= float(s.get("settle_s", -1)) <= 30:
            problems.append(f"{name}: window_s must be 1 to 60 and settle_s 0 to 30")
        unknown = set(s.get("engage", {})) - ENGAGE_KEYS
        if unknown:
            problems.append(f"{name}: engage keys {sorted(unknown)} are not scenario settings (the controller and the limits belong to the run)")
    return problems


def config_text(scenarios) -> str:
    return CFG_HEAD + "".join(f'bind {s["load_key"]} "load {s["save"]}"\n' for s in scenarios.values())


def install(game, scenarios, today) -> dict:
    """Write valve/ganglion.cfg and, once a day at most, copy valve/SAVE aside: staging writes
    saves, and a `map` start or an autosave trigger can overwrite the user's own."""
    valve = Path(game) / "valve"
    if not valve.is_dir():
        raise FileNotFoundError(f"{valve} is not a Half-Life valve directory")
    backup = valve / f"SAVE-backup-{today}"
    if (valve / "SAVE").is_dir() and not backup.exists():
        shutil.copytree(valve / "SAVE", backup)
    (valve / "ganglion.cfg").write_text(config_text(scenarios), encoding="ascii", newline="\n")
    return {"cfg": str(valve / "ganglion.cfg"), "backup": str(backup) if backup.exists() else None}


def check(game, scenario) -> dict:
    """Read a scenario's staged save offline: the state the trials will start from, and what is
    wrong with it. Hard skill is required: below it autoaim decides where the bullets go."""
    path = Path(game) / "valve" / "SAVE" / f"{scenario['save']}.sav"
    if not path.exists():
        return {"save": str(path), "problems": ["the save does not exist: stage the scenario first"], "staged": None}
    staged = saves.read(path)
    enemies = saves.alive(staged, scenario["enemy"])
    problems = []
    if staged["map"] != scenario["map"]:
        problems.append(f"map {staged['map']!r}, expected {scenario['map']!r}")
    if staged["skill"] != 3:
        problems.append(f"skill {staged['skill']}: stage at skill 3 (autoaim is off only there)")
    player = staged["player"]
    if player is None or not player["god"]:
        problems.append("god mode is not on in the save")
    if player is None or not player["notarget"]:
        problems.append("notarget is not on in the save")
    if len(enemies) < scenario["enemies"]:
        problems.append(f"{len(enemies)} {scenario['enemy']} alive, expected at least {scenario['enemies']}")
    return {"save": str(path), "problems": problems, "staged": staged,
            "enemy_health": sorted(m["health"] for m in enemies)}


def pair_order(seed, pairs, names) -> list[tuple[int, str]]:
    """(pair, condition) in running order: both conditions in every pair, their order drawn per pair."""
    rng = random.Random(seed)
    out = []
    for i in range(pairs):
        order = list(names)
        rng.shuffle(order)
        out += [(i, name) for name in order]
    return out


def rows_between(path, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(max(0, end - start))
    rows = []
    for line in data.decode("utf-8", "replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return sorted(rows, key=lambda e: e.get("t_mono", 0))


def fresh_save(path, before, *, timeout=6.0, sleep=time.sleep):
    """Wait for the game to finish writing the trial save: newer than `before`, size unchanged twice."""
    deadline, last = time.perf_counter() + timeout, None
    while time.perf_counter() < deadline:
        sleep(.25)
        if path.exists() and path.stat().st_mtime_ns != before:
            size = path.stat().st_size
            if size and size == last:
                return True
            last = size
    return False


def run_encounter(directory, scenario, controller, game, *, max_turn_px_s=1200.0, speed_px_s=1200.0, do=step, sleep=time.sleep):
    """One trial; no console, no weapon key, no walking. Returns the ledger's byte span and whether
    the closing save was written."""
    events = Path(directory) / "events.jsonl"
    trial_save = Path(game) / "valve" / "SAVE" / f"{TRIAL_SAVE}.sav"
    do(directory, {"op": "tap", "key": scenario["load_key"]})
    sleep(scenario["settle_s"])
    do(directory, {"op": "look"})
    start = events.stat().st_size if events.exists() else 0
    do(directory, {"op": "engage", "response": "track", "controller": controller, "max_turn_px_s": max_turn_px_s,
                   "speed_px_s": speed_px_s, **scenario["engage"]})
    if scenario.get("stimulus_key"):                     # back to back: nothing may fire at enemies that cannot see yet
        do(directory, {"op": "tap", "key": scenario["stimulus_key"]})
    do(directory, {"op": "wait", "s": scenario["window_s"]}, scenario["window_s"] + 10)
    do(directory, {"op": "disarm"})
    do(directory, {"op": "cancel"})
    before = trial_save.stat().st_mtime_ns if trial_save.exists() else None
    do(directory, {"op": "tap", "key": SAVE_KEY})
    saved = fresh_save(trial_save, before, sleep=sleep)
    do(directory, {"op": "look"})
    return {"start_byte": start, "end_byte": events.stat().st_size if events.exists() else 0, "saved": saved}


def record(rows, scenario, staged, after) -> dict:
    """A trial reduced to its outcome. Invalid, with the reasons, when the closing save is not the
    staged fight's, when the onset is not in the ledger, or when a shot went out before it."""
    why = []
    key = scenario.get("stimulus_key")
    onset_kind = "keys_held" if key else "reflex_armed"
    t0 = next((e["t_mono"] for e in rows if e["kind"] == onset_kind and (not key or e.get("keys") == [key])), None)
    if t0 is None:
        why.append("the onset is not in the ledger")
    fires = [e["t_mono"] for e in rows if e["kind"] == "input_submitted" and e.get("action") == "fire"]
    if t0 is not None and any(t < t0 for t in fires):
        why.append("fired before the stimulus")
    if after is None:
        why.append("the closing save was not written")
    else:
        why += saves.faults(staged, after, stimulus=bool(key))
    end = (t0 or 0) + scenario["window_s"]
    inside = [e for e in rows if t0 is not None and t0 <= e.get("t_mono", 0) <= end]
    shots = [e["t_mono"] for e in inside if e["kind"] == "input_submitted" and e.get("action") == "fire"]
    seen = [e["t_mono"] for e in inside if e["kind"] == "reflex_fired"]
    tolerance = float(scenario["engage"].get("tolerance_px", 14))
    near = [e["controller"] for e in inside if e["kind"] == "look_done" and e.get("error_px", 1e9) <= 3 * tolerance
            and e.get("controller") in ("connectome", "deterministic_override", "deterministic_stale")]
    result = saves.outcome(staged, after, scenario["enemy"]) if after is not None else {"enemies": None, "damage": None, "kills": None}
    return {"valid": not why, "why": why, **result,
            "detect_s": round(seen[0] - t0, 3) if seen else None,
            "first_shot_s": round(shots[0] - t0, 3) if shots else None,       # None: censored at the window
            "bursts": len(shots), "intents_before_onset": sum(1 for e in rows if e["kind"] == "intent_started" and t0 is not None and e["t_mono"] < t0),
            "near_model_share": round(near.count("connectome") / len(near), 3) if near else None,
            "summary": slice_summary(inside)}


def parse_conditions(text) -> dict:
    out = dict(part.split("=", 1) for part in text.split(","))
    if len(out) != 2 or not set(out.values()) <= {"deterministic", "connectome"}:
        raise ValueError("two conditions as name=controller, the controller deterministic or connectome")
    return out


def run(args):
    scenarios = load_scenarios(args.scenarios)
    scenario, conditions = scenarios[args.scenario], parse_conditions(args.conditions)
    if not scenario["frozen"] and set(conditions.values()) != {"deterministic"}:
        raise SystemExit(f"{args.scenario} is not frozen: its engage settings are still to be fixed from reference-only trials")
    state = check(args.game, scenario)
    if state["problems"]:
        raise SystemExit(f"{state['save']}: " + "; ".join(state["problems"]))
    directory, out = Path(args.dir), Path(args.out) / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    step(directory, {"op": "unwatch", "name": "flow"})
    step(directory, {"op": "tap", "key": CHEATS_KEY})
    block = {"label": args.label, "scenario": args.scenario, "spec": scenario, "conditions": conditions, "seed": args.seed,
             "max_turn_px_s": args.max_turn_px_s, "speed_px_s": args.speed_px_s, "staged_enemy_health": state["enemy_health"],
             "staged_skill": state["staged"]["skill"], "trials": []}
    trial_save = Path(args.game) / "valve" / "SAVE" / f"{TRIAL_SAVE}.sav"
    for order, (pair, name) in enumerate(pair_order(args.seed, args.pairs, list(conditions))):
        span = run_encounter(directory, scenario, conditions[name], args.game, max_turn_px_s=args.max_turn_px_s,
                             speed_px_s=args.speed_px_s)
        rows = rows_between(directory / "events.jsonl", span["start_byte"], span["end_byte"])
        trial = {"pair": pair, "order": order, "condition": name, "controller": conditions[name], **span,
                 **record(rows, scenario, state["staged"], saves.read(trial_save) if span["saved"] else None)}
        block["trials"].append(trial)
        out.write_text(json.dumps(block, indent=1), encoding="utf-8")
        print(json.dumps({k: trial[k] for k in ("pair", "condition", "valid", "why", "damage", "kills", "detect_s", "first_shot_s", "bursts")}), flush=True)
        if order == 0 and trial["summary"]["reflex_fired"] == 0:
            raise SystemExit("no engagement in the first trial: is ganglion.cfg exec'd, the save staged, the game in front?")
    print(f"wrote {out}")


def sign_flip_p(differences, *, samples=20000, seed=0) -> float | None:
    """Two-sided p of the mean under random sign flips: exact up to 16 differences, sampled beyond."""
    d = np.asarray([x for x in differences if x is not None], float)
    if len(d) == 0 or not np.any(d):
        return None
    observed = abs(d.mean())
    if len(d) <= 16:
        signs = np.array(list(product((1, -1), repeat=len(d))))
    else:
        signs = np.random.default_rng(seed).choice((1, -1), size=(samples, len(d)))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= observed - 1e-12))


def pair_differences(block, metric) -> list[float]:
    """Second condition minus first, for the pairs whose two trials are both valid. A first shot
    that never came counts as the window (censored there for both conditions alike)."""
    a, b = list(block["conditions"])
    window = block["spec"]["window_s"]
    by_pair = {}
    for t in block["trials"]:
        if t["valid"]:
            value = t[metric]
            by_pair.setdefault(t["pair"], {})[t["condition"]] = window if metric == "first_shot_s" and value is None else value
    return [p[b] - p[a] for p in by_pair.values() if a in p and b in p and p[a] is not None and p[b] is not None]


def analyse(args):
    """Per session: the pair differences of the primary and of the first shot, their median and a
    within-session sign-flip p (descriptive). Across sessions: the sign-flip p of the session
    medians, the only test a claim may rest on, and it needs five sessions to reach 0.0625."""
    sessions = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    names = {tuple(s["conditions"]) for s in sessions}
    if len(names) != 1 or len({s["scenario"] for s in sessions}) != 1:
        raise SystemExit("analyse one scenario and one pair of conditions at a time")
    a, b = names.pop()
    out = {"scenario": sessions[0]["scenario"], "difference": f"{b} minus {a}", "sessions": [], "across_sessions": {}}
    for metric in ("damage", "kills", "first_shot_s"):
        medians = []
        for s in sessions:
            d = pair_differences(s, metric)
            entry = next((e for e in out["sessions"] if e["label"] == s["label"]), None)
            if entry is None:
                valid = [t for t in s["trials"] if t["valid"]]
                entry = {"label": s["label"], "trials": len(s["trials"]), "invalid": len(s["trials"]) - len(valid),
                         "invalid_reasons": sorted({w for t in s["trials"] for w in t["why"]})}
                for name in (a, b):
                    mine = [t for t in valid if t["condition"] == name]
                    entry[name] = {"trials": len(mine), "damage_median": float(np.median([t["damage"] for t in mine])) if mine else None,
                                   "kills_mean": float(np.mean([t["kills"] for t in mine])) if mine else None,
                                   "first_shot_s_median": (float(np.median([t["first_shot_s"] if t["first_shot_s"] is not None else s["spec"]["window_s"] for t in mine]))
                                                           if mine else None),
                                   "fired_before_onset": sum("fired before the stimulus" in t["why"] for t in s["trials"] if t["condition"] == name)}
                out["sessions"].append(entry)
            entry[metric] = {"pairs": len(d), "median_difference": float(np.median(d)) if d else None, "p_within_session": sign_flip_p(d)}
            if d:
                medians.append(float(np.median(d)))
        out["across_sessions"][metric] = {"sessions": len(medians), "session_medians": medians,
                                          "p_sign_flip": sign_flip_p(medians) if len(medians) >= 2 else None}
    text = json.dumps(out, indent=1, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("install", "check", "run"):
        c = sub.add_parser(name)
        c.add_argument("--game", required=True, help="the Half-Life directory (the one holding valve/)")
        c.add_argument("--scenarios", default=SCENARIOS)
        if name != "install":
            c.add_argument("--scenario", required=True)
    r = sub.choices["run"]
    r.add_argument("--dir", required=True, help="the pilot's directory")
    r.add_argument("--out", default="runs/hl/encounters")
    r.add_argument("--label", required=True)
    r.add_argument("--pairs", type=int, default=15)
    r.add_argument("--seed", type=int, required=True, help="draws the order inside each pair; record one per session")
    r.add_argument("--conditions", default="reference=deterministic,model=connectome", help="name=controller,name=controller")
    r.add_argument("--max-turn-px-s", type=float, default=1200.0)
    r.add_argument("--speed-px-s", type=float, default=1200.0)
    a = sub.add_parser("analyse")
    a.add_argument("reports", nargs="+")
    a.add_argument("--out")
    args = p.parse_args()
    if args.cmd == "install":
        print(json.dumps(install(args.game, load_scenarios(args.scenarios), time.strftime("%Y%m%d")), indent=1))
    elif args.cmd == "check":
        state = check(args.game, load_scenarios(args.scenarios)[args.scenario])
        print(json.dumps({k: v for k, v in state.items() if k != "staged"}, indent=1))
        raise SystemExit(1 if state["problems"] else 0)
    else:
        (run if args.cmd == "run" else analyse)(args)


if __name__ == "__main__":
    main()
