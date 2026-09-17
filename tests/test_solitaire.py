import json
from pathlib import Path

import pytest

from ganglion.evaluation.solitaire.cards import Card, parse_cards as P
from ganglion.evaluation.solitaire.layout import CELL, COLUMN_X, COLUMN_Y, CARD_H, CARD_W, STRIP_H, Slot
from ganglion.evaluation.solitaire.player import boards_agree, deal_agrees, grab_region, landing
from ganglion.evaluation.solitaire.reader import Reader, Reading, Sighting
from ganglion.evaluation.solitaire.sawayama import (Board, Move, apply, autoplay, deduce, legal_moves, plan,
                                                    remember, run_start)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "ganglion/evaluation/solitaire/pack/sources.json"


def board(columns, waste="", stock=24, foundations=None, cell=None):
    return Board.build([P(c) for c in columns], P(waste), stock, foundations, Card.parse(cell) if cell else None, 3)


def test_reader_reads_every_labelled_frame_exactly():
    import cv2
    reader = Reader(learn=False)
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))["frames"]
    assert len(sources) >= 3
    for source in sources:
        frame = cv2.imread(str(ROOT / source["frame"]))
        reading = reader.read(frame)
        assert reading.problems == []
        for i, column in enumerate(source["columns"]):
            assert [str(s.card) for s in reading.columns[i]] == [str(c) for c in P(column)]
        assert [str(s.card) for s in reading.waste] == [str(c) for c in P(source["waste"])]
        for label, seen in zip(source.get("foundations", []), reading.foundations):
            assert (str(seen.card) if seen else "") == label
        assert all(s.method == "exact" for c in reading.columns for s in c)
        assert reading.stock_present and reading.stock == 52 - reading.visible


def test_rules_move_runs_and_refuse_illegal_targets():
    b = board(["Qs", "Kr 2d", "4s Jb Qc", "5d 9d 4h 4c", "3h Th 9s 3c 6s", "Ab 7c 8h 6h 5h 3d", "8s 3s 5c 9h Kb Qr Tc"])
    assert run_start(b.columns[6]) == 6 and run_start(b.columns[3]) == 3
    moves = {str(m) for m in legal_moves(b)}
    assert moves == {"c5[5] -> c3", "deal"}          # 3d onto 4c is the only tableau move
    after = apply(b, Move("column", "c5", 5, "c3"))
    assert [str(c) for c in after.columns[3]] == ["5d", "9d", "4h", "4c", "3d"] and len(after.columns[5]) == 5
    with pytest.raises(ValueError):
        apply(b, Move("column", "c0", 0, "c1"))        # Qs cannot go on 2d
    with pytest.raises(ValueError):
        apply(b, Move("column", "c6", 5, "c0"))        # not the start of a run


def test_deal_and_free_cell_follow_the_stock():
    b = board(["Qh", "Kr 2d", "", "", "", "", ""], stock=4)
    assert not b.cell_open and not any(m.kind == "cell" for m in legal_moves(b))
    dealt = apply(b, Move("deal", "stock", 0, "w"), tuple(P("9c 5h 6d")))
    assert dealt.stock == 1 and [str(c) for c in dealt.waste] == ["9c", "5h", "6d"]
    last = apply(dealt, Move("deal", "stock", 0, "w"), tuple(P("Jc")))
    assert last.stock == 0 and last.cell_open
    parked = apply(last, Move("cell", "w", 3, "cell"))
    assert str(parked.cell) == "Jc" and parked.waste == dealt.waste
    assert any(str(m) == "cell -> c0" for m in legal_moves(parked))   # Jc onto Qh
    with pytest.raises(ValueError):
        apply(dealt, Move("cell", "w", 2, "cell"))     # stock not yet empty


def test_the_game_auto_plays_aces_and_twos():
    b = board(["Ab", "2d 5h", "3d", "", "", "", ""], stock=0, foundations={"d": 1})
    settled = autoplay(b)
    # The game knows the suit of an exposed ace even when the reader does not: it goes up to
    # the first empty foundation of its colour, and the next reading confirms which.
    assert len(settled.columns[0]) == 0 and settled.foundations == (1, 0, 1, 0)
    known = board(["As", "5h 2d", "3d", "", "", "", ""], stock=0, foundations={"d": 1})
    settled = autoplay(known)
    assert settled.foundation_rank("s") == 1 and settled.foundation_rank("d") == 2
    assert [str(c) for c in settled.columns[2]] == ["3d"]           # threes wait for the opposite twos
    safe = board(["3d", "", "", "", "", "", ""], stock=0, foundations={"d": 2, "s": 2, "c": 2})
    assert autoplay(safe).foundation_rank("d") == 3


def test_remember_and_deduce_recover_hidden_suits():
    before = board(["Js", "6h Kh"], stock=0) if False else board(["Js", "6h Kh", "", "", "", "", ""], stock=0)
    seen = board(["Jb Td", "6h Kr", "", "", "", "", ""], stock=0)
    merged = remember(before, seen)
    assert str(merged.columns[0][0]) == "Js" and str(merged.columns[1][1]) == "Kh"
    partial = board(["Kb", "Ks", "Ar", "", "", "", ""], waste="", stock=0, foundations={"d": 1})
    deduced = deduce(partial)
    assert str(deduced.columns[0][0]) == "Kc" and str(deduced.columns[2][0]) == "Ah"


def test_planner_prefers_progress_and_reports_stuck():
    b = board(["Qs", "Kh 2d", "4s Jc 3d", "", "", "", ""], stock=0, foundations={"d": 2, "s": 0, "h": 0, "c": 0})
    p = plan(b)
    assert p.reason in ("improves", "win") and str(p.moves[0]) == "c2[2] -> f"
    stuck = board(["Kd", "Kc", "Kh", "Ks", "", "", ""], waste="", stock=0)
    assert plan(stuck).reason in ("stuck", "improves")
    assert plan(board(["Qs"] + [""] * 6, stock=3)).moves[0].kind == "deal"


def test_verification_tolerates_unknown_suits_and_auto_play():
    expected = board(["Js Td", "6h Kh", "", "", "", "", ""], stock=0)
    seen = board(["Jb Td", "6h Kr", "", "", "", "", ""], stock=0)
    assert boards_agree(expected, seen) and not boards_agree(expected, board(["Js", "6h Kh Td", "", "", "", "", ""], stock=0))
    before = board(["Js"] + [""] * 6, waste="9h", stock=3, foundations={"d": 0})
    after = board(["Js"] + [""] * 6, waste="9h 8d 8s", stock=0, foundations={"d": 1})   # Ad went up by itself
    assert deal_agrees(before, after, 3)
    assert not deal_agrees(before, board(["Js"] + [""] * 6, waste="9h 8d", stock=0, foundations={"d": 1}), 3)


def sighting(card, slot, covered):
    return Sighting(Card.parse(card), slot, covered, 1.0, 1.0, "exact")


def test_landing_geometry_targets_the_drop_and_a_band_the_face_will_cover():
    top = Slot(COLUMN_X[3], COLUMN_Y + 3 * 20, CARD_W, CARD_H)
    reading = Reading([[], [], [], [sighting("6s", top, False)], [], [], []], [], [None] * 4, False, None)
    slot, target, region = landing(reading, Move("column", "c6", 0, "c3"))
    assert (slot.x, slot.y) == (COLUMN_X[3], top.y + 20) and target == (slot.x, slot.y)
    assert region[1] == top.y + CARD_H + 4 and COLUMN_X[3] < region[0] and region[0] + region[2] < COLUMN_X[3] + CARD_W
    slot, target, region = landing(reading, Move("column", "c6", 0, "c0"))
    assert (slot.x, slot.y) == (COLUMN_X[0], COLUMN_Y)
    slot, target, region = landing(reading, Move("cell", "c3", 0, "cell"))
    assert (slot.x, slot.y) == CELL[:2]
    strip = grab_region(Slot(100, 200, CARD_W, STRIP_H))
    assert strip == (134, 201, 7, STRIP_H - 2)                        # between the pip columns
    assert grab_region(Slot(100, 200, CARD_W, CARD_H), 1) == (101, 201, 17, 14)               # rank-glyph margin
    assert grab_region(Slot(100, 200, CARD_W, CARD_H), 2) == (100 + CARD_W - 7, 201, 6, 30)   # right margin
    assert grab_region(Slot(100, 200, CARD_W, CARD_H), 3)[2] == 17
