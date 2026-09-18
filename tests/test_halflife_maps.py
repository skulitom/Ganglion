"""The BSP entity census: parsing, the start's facing, distances and bearings."""
import struct

import pytest

from ganglion.evaluation.halflife.maps import entities, hostiles, start_of


def bsp(text: str, version=30) -> bytes:
    lump = text.encode("latin-1")
    header = struct.pack("<i", version) + struct.pack("<ii", 124, len(lump)) + b"\0" * (14 * 8)
    return header + lump


MAP = '''
{
"classname" "worldspawn"
}
{
"classname" "info_player_start"
"origin" "100 100 0"
"angles" "0 90 0"
}
{
"classname" "monster_human_grunt"
"origin" "100 500 -64"
"spawnflags" "132"
"netname" "squad_a"
}
{
"classname" "monster_human_grunt"
"origin" "400 100 0"
}
{
"classname" "monster_scientist"
"origin" "110 110 0"
}
'''


def test_entities_are_read_from_the_lump_and_the_version_is_checked():
    ents = entities(bsp(MAP))
    assert [e["classname"] for e in ents][:2] == ["worldspawn", "info_player_start"]
    with pytest.raises(ValueError):
        entities(bsp(MAP, version=29))
    with pytest.raises(ValueError):
        entities(b"\0\0")


def test_hostiles_are_sorted_by_distance_with_bearing_relative_to_the_start_facing():
    ents = entities(bsp(MAP))
    assert start_of(ents) == ((100.0, 100.0, 0.0), 90.0)
    found = hostiles(ents)
    assert [h["distance"] for h in found] == [300, 405]                  # the scientist is not hostile
    right, ahead = found
    assert right["bearing_deg"] == -90 and right["flags"] == []          # +x is to the right of a start facing +y
    assert ahead["bearing_deg"] == 0 and ahead["dz"] == -64
    assert ahead["flags"] == ["monsterclip", "wait_for_script"] and ahead["squad"] == "squad_a"


def test_a_map_without_a_start_lists_nothing_and_a_bare_angle_is_a_yaw():
    assert hostiles(entities(bsp('{\n"classname" "monster_human_grunt"\n"origin" "0 0 0"\n}'))) == []
    ents = entities(bsp('{\n"classname" "info_player_start"\n"origin" "0 0 0"\n"angle" "180"\n}'))
    assert start_of(ents) == ((0.0, 0.0, 0.0), 180.0)
