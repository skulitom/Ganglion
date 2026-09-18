"""Encounter trials without the game: scenario rules, the key-only sequence, the record and the pairs."""
import json

import pytest

from ganglion.evaluation.halflife import encounters
from ganglion.evaluation.halflife.encounters import (check, config_text, install, load_scenarios, pair_differences,
                                                     pair_order, parse_conditions, record, run_encounter, sign_flip_p,
                                                     validate)
from ganglion.evaluation.halflife.saves import read
from test_halflife_saves import grunt, pack


def scenario(**over):
    base = {"map": "c2a2d", "save": "glc2a2d", "load_key": "f1", "stimulus_key": "f11", "enemy": "monster_human_grunt",
            "enemies": 3, "settle_s": 3.0, "window_s": 12.0, "primary": "damage", "engage": {"min_pixels": 150}, "frozen": False}
    return {**base, **over}


def test_the_shipped_scenarios_are_valid_and_trial_keys_are_function_keys():
    shipped = load_scenarios()
    assert "ledge-trio" in shipped and validate(shipped) == []
    for s in shipped.values():
        assert s["load_key"] not in encounters.FORBIDDEN_KEYS and s["load_key"] not in (encounters.SAVE_KEY, encounters.CHEATS_KEY)
    assert any("function key" in p for p in validate({"x": scenario(load_key="k")}))
    assert any("function key" in p for p in validate({"x": scenario(load_key="f6")}))          # the user's quicksave
    assert any("already loads" in p for p in validate({"x": scenario(), "y": scenario(save="glother")}))
    assert any("trial save" in p for p in validate({"x": scenario(save="gltrial")}))
    assert any("trial save" in p for p in validate({"x": scenario(save="_slot")}))
    assert any("only stimulus" in p for p in validate({"x": scenario(stimulus_key="f3")}))
    assert any("not scenario settings" in p for p in validate({"x": scenario(engage={"controller": "connectome"})}))
    assert any("no frozen" in p for p in validate({"x": {k: v for k, v in scenario().items() if k != "frozen"}}))
    assert validate({"x": scenario(stimulus_key=None)}) == []
    with pytest.raises(ValueError):
        parse_conditions("a=deterministic")
    with pytest.raises(ValueError):
        parse_conditions("a=deterministic,b=fast")
    assert parse_conditions("ref=deterministic,v6=connectome") == {"ref": "deterministic", "v6": "connectome"}


def test_install_writes_the_binds_and_copies_the_saves_aside_once(tmp_path):
    (tmp_path / "valve" / "SAVE").mkdir(parents=True)
    (tmp_path / "valve" / "SAVE" / "quick.sav").write_bytes(b"mine")
    done = install(tmp_path, {"x": scenario()}, "20260918")
    text = (tmp_path / "valve" / "ganglion.cfg").read_text(encoding="ascii")
    assert text == config_text({"x": scenario()})
    assert 'bind f1 "load glc2a2d"' in text and 'bind f4 "save gltrial"' in text and 'bind f11 "notarget"' in text
    assert "sv_cheats 1\"" in text and "\nsv_cheats 1" not in text          # a key, never a standing setting in the user's game
    assert (tmp_path / "valve" / "SAVE-backup-20260918" / "quick.sav").read_bytes() == b"mine" and done["backup"]
    (tmp_path / "valve" / "SAVE" / "quick.sav").write_bytes(b"later")
    install(tmp_path, {"x": scenario()}, "20260918")                         # the first copy of the day stands
    assert (tmp_path / "valve" / "SAVE-backup-20260918" / "quick.sav").read_bytes() == b"mine"
    with pytest.raises(FileNotFoundError):
        install(tmp_path / "nowhere", {}, "20260918")


def staged_save(tmp_path, **over):
    saves = tmp_path / "valve" / "SAVE"
    saves.mkdir(parents=True, exist_ok=True)
    player = {"classname": "player", "health": 100, "flags": 0xC8, "origin": (2800, 3052, -244)}
    (saves / "glc2a2d.sav").write_bytes(pack([{**player, **over.pop("player", {})}] + over.pop("grunts", [grunt(80)] * 3), **over))
    return saves


def test_check_accepts_only_a_hard_skill_save_with_god_notarget_and_the_enemies_alive(tmp_path):
    assert check(tmp_path, scenario())["problems"] == ["the save does not exist: stage the scenario first"]
    staged_save(tmp_path)
    ok = check(tmp_path, scenario())
    assert ok["problems"] == [] and ok["enemy_health"] == [80, 80, 80]
    staged_save(tmp_path, skill=1, player={"flags": 0x08}, grunts=[grunt(80), grunt(0, deadflag=2)], map_name="c1a3")
    bad = check(tmp_path, scenario())["problems"]
    assert len(bad) == 5 and any("autoaim" in p for p in bad) and any("1 monster_human_grunt alive" in p for p in bad)


def test_a_trial_is_keys_only_and_the_stimulus_follows_the_engage_at_once(tmp_path):
    saves = staged_save(tmp_path)
    (tmp_path / "events.jsonl").write_bytes(b"{}\n")
    calls = []

    def do(directory, command, timeout=30):
        calls.append(command)
        if command == {"op": "tap", "key": "f4"}:
            (saves / "gltrial.sav").write_bytes(b"x" * 64)
        return {}

    span = run_encounter(tmp_path, scenario(), "connectome", tmp_path, do=do, sleep=lambda s: None)
    ops = [(c["op"], c.get("key")) for c in calls]
    assert ops == [("tap", "f1"), ("look", None), ("engage", None), ("tap", "f11"), ("wait", None), ("disarm", None),
                   ("cancel", None), ("tap", "f4"), ("look", None)]
    assert all(c["op"] not in ("type", "walk", "fire", "turn") for c in calls)             # no console, no motion of the player
    engage = calls[2]
    assert engage["controller"] == "connectome" and engage["max_turn_px_s"] == engage["speed_px_s"] == 1200.0 and engage["min_pixels"] == 150
    assert span["saved"] and span["start_byte"] == 3 and calls[4]["s"] == 12.0
    calls.clear()
    run_encounter(tmp_path, scenario(stimulus_key=None), "deterministic", tmp_path, do=do, sleep=lambda s: None)
    assert ("tap", "f11") not in [(c["op"], c.get("key")) for c in calls]


def ledger(t0=100.0, *, fire_at=(1.4, 2.2), seen_at=.35, early_fire=False, key="f11"):
    rows = [{"kind": "reflex_armed", "t_mono": t0 - .2}, {"kind": "keys_held", "t_mono": t0, "keys": [key]},
            {"kind": "keys_held", "t_mono": t0 + 13, "keys": ["f4"]}]
    if early_fire:
        rows.append({"kind": "input_submitted", "action": "fire", "t_mono": t0 - .1, "intent_id": "i0"})
    rows.append({"kind": "reflex_fired", "t_mono": t0 + seen_at})
    for t in fire_at:
        rows.append({"kind": "input_submitted", "action": "fire", "t_mono": t0 + t, "intent_id": "i1"})
    for i in range(10):
        rows.append({"kind": "look_done", "t_mono": t0 + .5 + i * .01, "percept_ready_mono": t0 + .5 + i * .01, "delta": [5, 0],
                     "error_px": 60 - 5 * i, "intent_id": "i1", "controller": "connectome" if i % 2 else "deterministic_override"})
    rows.append({"kind": "input_submitted", "action": "fire", "t_mono": t0 + 12.5, "intent_id": "i1"})      # after the window
    return sorted(rows, key=lambda e: e["t_mono"])


def test_record_scores_a_trial_from_the_save_and_names_what_invalidates_it(tmp_path):
    saves = staged_save(tmp_path)
    staged = read(saves / "glc2a2d.sav")
    (saves / "after.sav").write_bytes(pack([{"classname": "player", "health": 100, "flags": 0x48, "origin": (2801, 3052, -244)},
                                            grunt(80), grunt(25), grunt(-5, deadflag=2)]))
    after = read(saves / "after.sav")
    r = record(ledger(), scenario(), staged, after)
    assert r["valid"] and r["why"] == [] and r["damage"] == 135.0 and r["kills"] == 1 and r["enemies"] == 3
    assert r["detect_s"] == .35 and r["first_shot_s"] == 1.4 and r["bursts"] == 2          # the shot after the window is not counted
    assert r["near_model_share"] == .5                       # steps within three tolerances: 40 px and under, half from the model
    never = record(ledger(fire_at=()), scenario(), staged, after)
    assert never["valid"] and never["first_shot_s"] is None and never["bursts"] == 0
    early = record(ledger(early_fire=True), scenario(), staged, after)
    assert not early["valid"] and early["why"] == ["fired before the stimulus"]
    assert record(ledger(key="f9"), scenario(), staged, after)["why"] == ["the onset is not in the ledger"]
    unsaved = record(ledger(), scenario(), staged, None)
    assert not unsaved["valid"] and unsaved["damage"] is None and "the closing save was not written" in unsaved["why"]
    (saves / "unseen.sav").write_bytes(pack([{"classname": "player", "health": 100, "flags": 0xC8, "origin": (2800, 3052, -244)}] + [grunt(80)] * 3))
    assert any("notarget is still on" in w for w in record(ledger(), scenario(), staged, read(saves / "unseen.sav"))["why"])
    # without a stimulus the onset is the arming of the reflex and notarget stays on
    quiet = record(ledger(), scenario(stimulus_key=None), staged, read(saves / "unseen.sav"))
    assert quiet["valid"] and quiet["first_shot_s"] == 1.6
    json.dumps(r, allow_nan=False)


def test_pairs_hold_both_conditions_in_a_seeded_order():
    order = pair_order(7, 20, ["a", "b"])
    assert order == pair_order(7, 20, ["a", "b"]) and order != pair_order(8, 20, ["a", "b"])
    assert [p for p, _ in order] == [i for i in range(20) for _ in range(2)]
    firsts = [name for i, (_, name) in enumerate(order) if i % 2 == 0]
    assert all({order[2 * i][1], order[2 * i + 1][1]} == {"a", "b"} for i in range(20)) and 3 < firsts.count("a") < 17


def test_differences_are_taken_inside_valid_pairs_and_the_sign_flip_test_is_exact():
    block = {"conditions": {"ref": "deterministic", "v6": "connectome"}, "spec": {"window_s": 12.0}, "trials": [
        {"pair": 0, "condition": "ref", "valid": True, "damage": 100, "first_shot_s": 2.0},
        {"pair": 0, "condition": "v6", "valid": True, "damage": 140, "first_shot_s": None},            # censored at the window
        {"pair": 1, "condition": "v6", "valid": True, "damage": 90, "first_shot_s": 1.0},
        {"pair": 1, "condition": "ref", "valid": False, "damage": 300, "first_shot_s": .1},            # the pair is dropped
        {"pair": 2, "condition": "ref", "valid": True, "damage": 80, "first_shot_s": 1.5},
        {"pair": 2, "condition": "v6", "valid": True, "damage": 70, "first_shot_s": 1.25}]}
    assert pair_differences(block, "damage") == [40, -10]
    assert pair_differences(block, "first_shot_s") == [10.0, -.25]
    assert sign_flip_p([1, 1, 1, 1, 1]) == 2 / 32                     # all one sign: the two extreme flips of 32
    assert sign_flip_p([1, -1]) == 1.0 and sign_flip_p([]) is None and sign_flip_p([0, 0]) is None
    assert sign_flip_p([3, 2, 4, 1, 5, 2, 3, 4, 1, 2, 3, 4, 2, 1, 3, 2, 4, 1]) < .001       # sampled beyond sixteen
