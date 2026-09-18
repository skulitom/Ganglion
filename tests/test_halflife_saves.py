"""The save reader: a hand-packed save gives health, flags, damage, kills and the trial's faults."""
import struct

import pytest

from ganglion.evaluation.halflife.saves import blocks, faults, outcome, read


def pack(entities, *, map_name="c2a2d", skill=3):
    """A minimal save: some container bytes, then a VALV section with a token table and blocks."""
    tokens = ["", "Save Header", "mapName", "skillLevel", "ENTVARS", "classname", "health", "deadflag", "flags", "origin", "v_angle"]
    index = {t: i for i, t in enumerate(tokens)}

    def field(name, raw):
        return struct.pack("<hH", len(raw), index[name]) + raw

    def block(name, fields):
        return struct.pack("<hHi", 4, index[name], len(fields)) + b"".join(fields)

    body = block("Save Header", [field("mapName", map_name.encode() + b"\0"), field("skillLevel", struct.pack("<i", skill))])
    for e in entities:
        fields = [field("classname", e["classname"].encode() + b"\0"), field("health", struct.pack("<f", e["health"])),
                  field("deadflag", struct.pack("<f", e.get("deadflag", 0))), field("flags", struct.pack("<i", e.get("flags", 0))),
                  field("origin", struct.pack("<3f", *e.get("origin", (0, 0, 0))))]
        if e["classname"] == "player":
            fields.append(field("v_angle", struct.pack("<3f", 0, e.get("yaw", 0), 0)))
        body += block("ENTVARS", fields)
    table = b"\0".join(t.encode() for t in tokens) + b"\0"
    return b"JSAV" + b"\x71\0\0\0" * 3 + b"VALV" + struct.pack("<5i", 113, len(body), 1, len(tokens), len(table)) + table + body


def grunt(health, **kw):
    return {"classname": "monster_human_grunt", "health": health, **kw}


def test_a_save_is_read_into_the_player_and_the_monsters(tmp_path):
    path = tmp_path / "staged.sav"
    path.write_bytes(pack([{"classname": "player", "health": 100, "flags": 0x248 | 0x80, "origin": (2800, 3052, -244), "yaw": 274},
                           grunt(80, origin=(2850, 2300, -172)), grunt(80), grunt(0, deadflag=2),
                           {"classname": "monster_headcrab", "health": 10}, {"classname": "func_door", "health": 0}]))
    s = read(path)
    assert s["map"] == "c2a2d" and s["skill"] == 3
    assert s["player"]["god"] and s["player"]["notarget"] and s["player"]["yaw"] == 274 and s["player"]["origin"] == [2800, 3052, -244]
    assert [m["classname"] for m in s["monsters"]] == ["monster_human_grunt"] * 3 + ["monster_headcrab"]
    assert s["monsters"][0]["origin"] == [2850, 2300, -172] and s["monsters"][2]["dead"]
    assert [name for name, _ in blocks(path.read_bytes())][:2] == ["Save Header", "ENTVARS"]
    with pytest.raises(ValueError):
        blocks(b"not a save")


def test_outcome_is_the_fall_in_total_health_and_the_kills(tmp_path):
    (tmp_path / "a.sav").write_bytes(pack([{"classname": "player", "health": 100, "flags": 0xC8}, grunt(80), grunt(80), grunt(80),
                                           {"classname": "monster_headcrab", "health": 10}]))
    (tmp_path / "b.sav").write_bytes(pack([{"classname": "player", "health": 100, "flags": 0x48}, grunt(35), grunt(-12, deadflag=2),
                                           {"classname": "monster_headcrab", "health": 0, "deadflag": 2}]))
    staged, after = read(tmp_path / "a.sav"), read(tmp_path / "b.sav")
    # one wounded for 45, one dead in the save, one gone from it: 45 + 80 + 80
    assert outcome(staged, after) == {"enemies": 3, "damage": 205.0, "kills": 2}
    assert outcome(staged, staged) == {"enemies": 3, "damage": 0.0, "kills": 0}
    assert outcome(staged, after, "monster_headcrab") == {"enemies": 1, "damage": 10.0, "kills": 1}
    assert faults(staged, after) == []


def test_faults_name_what_makes_a_trial_not_the_staged_fight(tmp_path):
    (tmp_path / "a.sav").write_bytes(pack([{"classname": "player", "health": 100, "flags": 0xC0, "origin": (0, 0, 0)}, grunt(80)]))
    staged = read(tmp_path / "a.sav")
    (tmp_path / "b.sav").write_bytes(pack([{"classname": "player", "health": 100, "flags": 0x80, "origin": (90, 0, 0)}, grunt(80)],
                                          map_name="c1a3", skill=1))
    why = faults(staged, read(tmp_path / "b.sav"))
    assert len(why) == 5 and "god mode is off" in why and any("notarget is still on" in w for w in why)
    assert any("moved 90 units" in w for w in why) and any("c1a3" in w for w in why) and any("skill 1" in w for w in why)
    # a scenario without a stimulus keeps notarget on by design
    (tmp_path / "c.sav").write_bytes(pack([{"classname": "player", "health": 100, "flags": 0xC0, "origin": (10, 0, 0)}, grunt(80)]))
    assert faults(staged, read(tmp_path / "c.sav"), stimulus=False) == []
    (tmp_path / "d.sav").write_bytes(pack([grunt(80)]))
    assert faults(staged, read(tmp_path / "d.sav")) == ["no player in the save"]
