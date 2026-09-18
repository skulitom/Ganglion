"""What a save file says happened: every monster's health and the player's flags, read offline.

A GoldSrc .sav is an uncompressed stream of token-named fields. The entity data starts at the
first "VALV" tag: a header of five ints, a table of zero-terminated token strings, then blocks
of [short 4][token][int fields] followed by fields of [short size][token][bytes]. ENTVARS blocks
carry classname, health, deadflag, flags and origin for every entity, the "Save Header" block the
map and the skill. A key bound to `save <name>` at the end of a trial therefore gives the damage
dealt and the kills without the console, a log or a guess from pixels, and the same file proves
the trial was the staged one (god mode still on, the stimulus delivered, the player in place).

    python -m ganglion.evaluation.halflife.saves path/to/valve/SAVE/quick.sav
"""
from __future__ import annotations

import argparse
import json
from math import dist
from pathlib import Path
import struct

FL_GODMODE, FL_NOTARGET = 0x40, 0x80


def blocks(data: bytes) -> list[tuple[str, dict]]:
    """The (name, {field: bytes}) blocks of the first VALV section."""
    at = data.find(b"VALV")
    if at < 0:
        raise ValueError("not a GoldSrc save: no VALV section")
    _version, _size, _tables, token_count, token_size = struct.unpack_from("<5i", data, at + 4)
    p = at + 24
    tokens = [t.decode("latin-1") for t in data[p:p + token_size].split(b"\0")[:token_count]]
    p += token_size
    out = []
    while p + 8 <= len(data):
        size, token = struct.unpack_from("<hH", data, p)
        if size != 4 or token >= len(tokens):
            break
        count = struct.unpack_from("<i", data, p + 4)[0]
        p += 8
        fields = {}
        for _ in range(count):
            if p + 4 > len(data):
                return out
            field_size, field_token = struct.unpack_from("<hH", data, p)
            if field_size < 0 or field_token >= len(tokens):
                return out
            fields[tokens[field_token]] = data[p + 4:p + 4 + field_size]
            p += 4 + field_size
        out.append((tokens[token], fields))
    return out


def _text(raw):
    return raw.split(b"\0")[0].decode("latin-1")


def _floats(raw, n=1):
    return struct.unpack(f"<{n}f", raw[:4 * n]) if raw is not None and len(raw) >= 4 * n else None


def read(path) -> dict:
    """{"map", "skill", "player": {...}, "monsters": [...]} of a save. `flags` is the raw entity
    flag word (the SDK saves the int through a float-sized field, bytes unchanged)."""
    found = blocks(Path(path).read_bytes())
    header = next((f for name, f in found if name == "Save Header"), {})
    out = {"map": _text(header.get("mapName", b"")), "player": None, "monsters": [],
           "skill": struct.unpack("<i", header["skillLevel"][:4])[0] if len(header.get("skillLevel", b"")) >= 4 else None}
    for name, f in found:
        if name != "ENTVARS":
            continue
        classname = _text(f.get("classname", b""))
        if classname != "player" and not classname.startswith("monster_"):
            continue
        health, dead, origin = _floats(f.get("health")), _floats(f.get("deadflag")), _floats(f.get("origin"), 3)
        entity = {"classname": classname, "health": health[0] if health else 0.0, "dead": bool(dead and dead[0]),
                  "flags": struct.unpack("<i", f["flags"][:4])[0] if len(f.get("flags", b"")) >= 4 else 0,
                  "origin": [round(v, 1) for v in origin] if origin else None}
        if classname == "player":
            view = _floats(f.get("v_angle"), 3)
            entity.update(god=bool(entity["flags"] & FL_GODMODE), notarget=bool(entity["flags"] & FL_NOTARGET),
                          yaw=round(view[1], 1) if view else 0.0)          # a save omits fields that are zero
            out["player"] = entity
        else:
            out["monsters"].append(entity)
    return out


def alive(save, classname):
    return [m for m in save["monsters"] if m["classname"] == classname and not m["dead"] and m["health"] > 0]


def outcome(staged, after, classname="monster_human_grunt") -> dict:
    """Damage dealt and kills between the staged save and the one written at the end of a trial:
    the fall in the class's total health (a dead or missing monster counts as zero), so no
    monster has to be matched with its earlier self."""
    before, now = alive(staged, classname), alive(after, classname)
    return {"enemies": len(before), "damage": round(sum(m["health"] for m in before) - sum(m["health"] for m in now), 1),
            "kills": len(before) - len(now)}


def faults(staged, after, *, stimulus=True, moved_limit=40.0) -> list[str]:
    """Why a trial's closing save does not describe the staged fight; [] when it does."""
    why = []
    if after["map"] != staged["map"]:
        why.append(f"map {after['map']!r}, staged on {staged['map']!r}")
    if after["skill"] != staged["skill"]:
        why.append(f"skill {after['skill']}, staged at {staged['skill']}")
    a, b = staged["player"], after["player"]
    if a is None or b is None:
        return why + ["no player in the save"]
    if not b["god"]:
        why.append("god mode is off")
    if stimulus and b["notarget"]:
        why.append("the stimulus did not arrive: notarget is still on")
    if a["origin"] and b["origin"] and dist(a["origin"], b["origin"]) > moved_limit:
        why.append(f"the player moved {dist(a['origin'], b['origin']):.0f} units")
    return why


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("save", type=Path)
    p.add_argument("--classname", default="monster_human_grunt")
    a = p.parse_args()
    s = read(a.save)
    s["monsters"] = [m for m in s["monsters"] if m["classname"] == a.classname]
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
