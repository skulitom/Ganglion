"""Where the enemies stand, read from the maps themselves.

A GoldSrc BSP carries its entities as text (lump 0). For choosing encounters that start in view of
the player this lists, per map, the hostile monsters by distance from the `info_player_start` a
direct `map <name>` gives, with the height difference, the bearing relative to the start's facing
(the view is 90 degrees wide, so within 45 is on screen if nothing is in the way; line of sight is
not in the entity lump) and the spawn flags that say whether a monster waits for a script. A
census is where to look, not what is there: hardly any map shows its grunts from the start, and
every encounter worth staging needs a walk (see docs/bench/HALFLIFE.md).

    python -m ganglion.evaluation.halflife.maps <game>/valve/maps c2a2d c2a4e
"""
from __future__ import annotations

import argparse
import json
from math import atan2, degrees, dist
from pathlib import Path
import re
import struct

BSP_VERSION = 30
HOSTILE = ("monster_human_grunt", "monster_grunt_repel", "monster_human_assassin", "monster_alien_slave",
           "monster_alien_grunt", "monster_alien_controller", "monster_headcrab", "monster_zombie", "monster_houndeye",
           "monster_bullchicken", "monster_sentry", "monster_miniturret", "monster_turret")
FLAGS = {1: "wait_till_seen", 2: "gag", 4: "monsterclip", 16: "prisoner", 32: "squad_leader", 128: "wait_for_script",
         256: "pre_disaster", 512: "fade_corpse"}


def entities(data: bytes) -> list[dict]:
    """The entity lump of a version 30 BSP as a list of key/value dicts (later duplicate keys win)."""
    if len(data) < 12:
        raise ValueError("not a BSP: too short")
    version, offset, length = struct.unpack_from("<iii", data, 0)
    if version != BSP_VERSION:
        raise ValueError(f"BSP version {version}, expected {BSP_VERSION}")
    text = data[offset:offset + length].decode("latin-1", "replace")
    return [dict(re.findall(r'"([^"]*)"\s+"([^"]*)"', block)) for block in re.findall(r"\{([^{}]*)\}", text)]


def vector(text) -> tuple | None:
    try:
        v = tuple(float(x) for x in str(text).split())
    except ValueError:
        return None
    return v if len(v) == 3 else None


def start_of(ents) -> tuple[tuple, float] | None:
    """Origin and yaw of the first info_player_start: `angles` is "pitch yaw roll", `angle` a bare yaw."""
    for e in ents:
        if e.get("classname") == "info_player_start" and vector(e.get("origin")):
            angles = vector(e.get("angles"))
            yaw = angles[1] if angles else float(e.get("angle", 0) or 0)
            return vector(e["origin"]), yaw
    return None


def hostiles(ents, classes=HOSTILE) -> list[dict]:
    """Hostile monsters sorted by distance from the map's start; [] when the map has no start."""
    start = start_of(ents)
    if start is None:
        return []
    origin, yaw = start
    out = []
    for e in ents:
        at = vector(e.get("origin"))
        if e.get("classname") not in classes or at is None:
            continue
        flags = int(e.get("spawnflags", 0) or 0)
        out.append({"classname": e["classname"], "distance": round(dist(origin, at)), "dz": round(at[2] - origin[2]),
                    "bearing_deg": round((degrees(atan2(at[1] - origin[1], at[0] - origin[0])) - yaw + 180) % 360 - 180),
                    "flags": [name for bit, name in FLAGS.items() if flags & bit],
                    "targetname": e.get("targetname"), "squad": e.get("netname")})
    return sorted(out, key=lambda h: h["distance"])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("maps", help="the game's maps directory")
    p.add_argument("names", nargs="*", help="map names (default: every map with a hostile within --within units)")
    p.add_argument("--within", type=float, default=1500, help="list hostiles up to this distance from the start")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    report = {}
    for path in sorted(Path(a.maps).glob("*.bsp")):
        if a.names and path.stem not in a.names:
            continue
        try:
            near = [h for h in hostiles(entities(path.read_bytes())) if h["distance"] <= a.within]
        except ValueError:
            continue
        if near:
            report[path.stem] = near
    if a.json:
        print(json.dumps(report, indent=1))
        return
    for name, near in report.items():
        print(name)
        for h in near:
            print(f"  {h['distance']:5d} u  {h['classname'][8:]:16s} dz {h['dz']:5d}  bearing {h['bearing_deg']:5d}  "
                  f"{','.join(h['flags']) or '-':28s} {h['squad'] or h['targetname'] or ''}")


if __name__ == "__main__":
    main()
