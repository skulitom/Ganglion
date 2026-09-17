"""Pixel geometry of the Sawayama window at its 1280x720 client size.

Measured from a captured frame on 2026-09-17 (build 24998607). All rectangles are
[x, y, width, height] in client pixels of the game window; the player adds the window origin.
"""
from __future__ import annotations

from dataclasses import dataclass

CLIENT = (1280, 720)

CARD_W, CARD_H = 73, 99          # white face interior of a fully visible card
STRIP_H = 16                      # visible white interior of a covered tableau card
PITCH_Y = 20                      # vertical offset between stacked tableau cards
COLUMN_X = (401, 488, 575, 661, 748, 835, 921)
COLUMN_Y = 284                    # interior top of the first tableau card
RANK_W = 20                       # rank glyph lives in the strip's first 20 columns
STOCK = (401, 156, 73, 99)        # face-down deck; empties into the free cell
CELL = (401, 171, 73, 99)         # a card parked in the emptied stock slot sits 15 px lower than the deck
CELL_SEARCH = (393, 150, 90, 125) # region searched for a parked card
FOUNDATIONS = ((283, 154, 74, 101), (283, 274, 74, 101), (283, 394, 74, 101), (283, 514, 74, 101))
WASTE_X = 488                     # first waste card interior x (fanned rightwards)
WASTE_Y = 171
WASTE_PITCH = 19                  # nominal fan pitch (18-19 px, fractional); corners stay visible
WASTE_BAND = (480, 165, 540, 110)  # region searched for waste card faces

CARD_WHITE_BGR = (245, 250, 250)


@dataclass(frozen=True)
class Slot:
    """A card place on screen: interior rectangle and the point to grab or drop at."""
    x: int
    y: int
    w: int
    h: int

    @property
    def rect(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def centre(self):
        return (self.x + self.w // 2, self.y + self.h // 2)


def column_card(column: int, index: int, count: int) -> Slot:
    """Interior of the index-th card in a column holding count cards."""
    y = COLUMN_Y + PITCH_Y * index
    visible = CARD_H if index == count - 1 else STRIP_H
    return Slot(COLUMN_X[column], y, CARD_W, visible)


def column_landing(column: int, count: int) -> Slot:
    """Strip a card dropped on this column will occupy."""
    return Slot(COLUMN_X[column], COLUMN_Y + PITCH_Y * count, CARD_W, STRIP_H)


def waste_card(index: int, count: int) -> Slot:
    x = WASTE_X + WASTE_PITCH * index
    visible = CARD_W if index == count - 1 else WASTE_PITCH
    return Slot(x, WASTE_Y, visible, CARD_H)


def foundation(slot: int) -> Slot:
    return Slot(*FOUNDATIONS[slot])


def stock() -> Slot:
    return Slot(*STOCK)
