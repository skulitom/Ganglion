"""Read a Sawayama board from a captured frame using the taught template pack.

The game scales its pixel art by 4/3, so a glyph's exact pixels depend on where the card sits:
its phase is (3x mod 4, 3y mod 4). Templates are stored per phase and matched exactly; a phase
that has not been taught falls back to normalised correlation with a required margin, and a
confident fallback is recorded as that phase's template. Failures are explicit, never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ganglion.percepts.templates import ink_bbox, ink_mask, iou, locate, mean_ink_colour
from .cards import Card, RANKS, SUITS
from .layout import (CARD_H, CARD_W, CELL_SEARCH, COLUMN_X, COLUMN_Y, FOUNDATIONS, PITCH_Y, RANK_W, STOCK,
                     STRIP_H, WASTE_BAND, Slot)
from .sawayama import Board

PACK = Path(__file__).with_name("pack")
CORNER_H = 32           # rank glyph rows 0..15, suit glyph rows 16..31
EXACT = 0.9             # aligned IoU accepted as an exact phase match
FALLBACK = 0.65         # minimum normalised correlation for an untaught phase
MARGIN = 0.06           # lead over the best other label required in fallback
LEARN_SCORE, LEARN_MARGIN = 0.8, 0.1   # a fallback this confident becomes the phase's template
PIP_MATCH = 0.7


def phase(x: int, y: int) -> str:
    return f"{(3 * x) % 4}{(3 * y) % 4}"


def _white(bgr):
    return (bgr[..., 0] > 225) & (bgr[..., 1] > 225) & (bgr[..., 2] > 225)


def _tan(bgr):
    b, g, r = (bgr[..., i].astype(int) for i in range(3))
    return (r > 180) & (g > 150) & (b > 100) & (r - b > 40)


def is_red(colour) -> bool:
    b, g, r = colour
    return r - b > 80


def card_ink(bgr: np.ndarray) -> np.ndarray:
    """uint8 mask of index-glyph ink: black or red print only. The tan of card borders and of
    the large ace pip is excluded, so a corner region holds nothing but its rank and suit glyphs."""
    b, g, r = (bgr[..., i].astype(int) for i in range(3))
    dark = np.maximum(np.maximum(b, g), r) < 140
    red = (r > 150) & (r - g > 60) & (r - b > 60)
    return np.where(dark | red, 255, 0).astype(np.uint8)


def tight(mask: np.ndarray) -> np.ndarray:
    box = ink_bbox(mask)
    if box is None:
        return mask[:0, :0]
    x, y, w, h = box
    return mask[y:y + h, x:x + w]


def aligned_iou(a: np.ndarray, b: np.ndarray) -> float:
    ta, tb = tight(a), tight(b)
    if ta.size == 0 or tb.size == 0:
        return 0.0
    h, w = max(ta.shape[0], tb.shape[0]), max(ta.shape[1], tb.shape[1])
    pa, pb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    pa[:ta.shape[0], :ta.shape[1]], pb[:tb.shape[0], :tb.shape[1]] = ta, tb
    return iou(pa, pb)


class Glyphs:
    """Templates for one glyph family: label -> phase -> ink mask."""

    def __init__(self, directory: Path, labels: str):
        import cv2
        self.directory, self.labels = Path(directory), labels
        self.templates: dict[str, dict[str, np.ndarray]] = {l: {} for l in labels}
        self.widths: dict[str, dict[str, int]] = {l: {} for l in labels}
        for file in sorted(self.directory.glob("*.png")):
            label, _, rest = file.stem.partition("_")
            key, _, width = rest.partition("_w")
            if label in self.templates:
                mask = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                self.templates[label][key] = mask
                self.widths[label][key] = int(width) if width else mask.shape[1]

    def known(self) -> list[str]:
        return [l for l, v in self.templates.items() if v]

    def record(self, label: str, key: str, mask: np.ndarray, width: int | None = None):
        """Store a phase template; a partially visible glyph records its visible width."""
        import cv2
        width = mask.shape[1] if width is None else width
        old = self.widths[label].get(key)
        if old is not None and old >= width:
            return
        self.templates[label][key] = mask.copy()
        self.widths[label][key] = width
        self.directory.mkdir(parents=True, exist_ok=True)
        for stale in self.directory.glob(f"{label}_{key}*.png"):
            stale.unlink()
        suffix = "" if width >= mask.shape[1] else f"_w{width}"
        cv2.imwrite(str(self.directory / f"{label}_{key}{suffix}.png"), mask)

    def classify(self, mask: np.ndarray, key: str, *, columns: int | None = None, learn: bool,
                 labels: str | None = None) -> tuple[str, float, str]:
        """(label, score, method). Raises ValueError when nothing is confidently readable."""
        visible = mask.shape[1] if columns is None else columns
        view = mask[:, :visible]
        allowed = {l: v for l, v in self.templates.items() if labels is None or l in labels}
        exact = {}
        for label, variants in allowed.items():
            template = variants.get(key)
            if template is not None:
                shared = min(visible, self.widths[label][key])
                exact[label] = aligned_iou(view[:, :shared], template[:, :shared])
        if exact:
            label = max(exact, key=exact.get)
            if exact[label] >= EXACT:
                return label, exact[label], "exact"
        scores = {}
        query = tight(view)
        for label, variants in allowed.items():
            best = 0.0
            for variant, template in variants.items():
                probe = tight(template[:, :min(visible, self.widths[label][variant])])
                if probe.size == 0 or query.size == 0:
                    continue
                # Correlate the smaller crop inside the larger one, tolerant of a pixel of drift.
                a, b = (query, probe) if query.size >= probe.size else (probe, query)
                pad = np.zeros((a.shape[0] + 4, a.shape[1] + 4), np.uint8)
                pad[2:2 + a.shape[0], 2:2 + a.shape[1]] = a
                best = max(best, locate(pad, b)[0])
            scores[label] = best
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        if not ordered or ordered[0][1] < FALLBACK:
            raise ValueError(f"no template matches (best {ordered[0] if ordered else None})")
        label, score = ordered[0]
        runner = ordered[1][1] if len(ordered) > 1 else 0.0
        if score - runner < MARGIN:
            raise ValueError(f"ambiguous between {label} {score:.2f} and {ordered[1][0]} {runner:.2f}")
        if learn and score >= LEARN_SCORE and score - runner >= LEARN_MARGIN:
            self.record(label, key, mask, visible)
        return label, score, "fallback"


@dataclass
class Pack:
    ranks: Glyphs
    suits: Glyphs
    pips: Glyphs

    @classmethod
    def load(cls, directory: Path = PACK) -> "Pack":
        directory = Path(directory)
        pack = cls(Glyphs(directory / "ranks", RANKS), Glyphs(directory / "suits", SUITS), Glyphs(directory / "pips", SUITS))
        missing = [n for n, g in (("ranks", pack.ranks), ("suits", pack.suits), ("pips", pack.pips)) if len(g.known()) < len(g.labels)]
        if missing:
            raise ValueError(f"template pack {directory} lacks labels in {missing}")
        return pack


@dataclass
class Sighting:
    card: Card
    slot: Slot                      # visible interior in client pixels
    covered: bool
    rank_score: float
    suit_score: float
    method: str
    notes: list[str] = field(default_factory=list)


@dataclass
class Reading:
    columns: list[list[Sighting]]
    waste: list[Sighting]
    foundations: list[Sighting | None]
    stock_present: bool
    cell: Sighting | None
    problems: list[str] = field(default_factory=list)

    @property
    def visible(self) -> int:
        return (sum(len(c) for c in self.columns) + len(self.waste) + (self.cell is not None)
                + sum(s.card.rank for s in self.foundations if s))

    @property
    def stock(self) -> int:
        return 52 - self.visible if self.stock_present else 0

    @property
    def readable(self) -> bool:
        return not self.problems

    def foundation_suits(self) -> list[str | None]:
        return [s.card.suit if s else None for s in self.foundations]

    def board(self, deal_count: int) -> Board:
        found = {}
        for s in self.foundations:
            if s:
                found[s.card.suit] = s.card.rank
        return Board.build([[s.card for s in c] for c in self.columns],
                           waste=[s.card for s in self.waste], stock=self.stock,
                           foundations=found, cell=self.cell.card if self.cell else None,
                           deal_count=deal_count)

    def describe(self) -> str:
        lines = [f"stock {'present' if self.stock_present else 'empty'} ({self.stock} inferred)  "
                 f"waste [{' '.join(str(s.card) for s in self.waste)}]  "
                 f"foundations {[str(s.card) if s else '-' for s in self.foundations]}  "
                 f"cell {self.cell.card if self.cell else '-'}"]
        for i, column in enumerate(self.columns):
            lines.append(f"c{i}: {' '.join(str(s.card) for s in column)}")
        if self.problems:
            lines.append("problems: " + "; ".join(self.problems))
        return "\n".join(lines)


class Reader:
    def __init__(self, pack: Pack | None = None, *, learn: bool = True):
        self.pack = pack or Pack.load()
        self.learn = learn

    # -- primitives ---------------------------------------------------------------------------
    def strip_at(self, frame, x, y) -> bool:
        if y < 2 or y + STRIP_H > frame.shape[0] or x + CARD_W > frame.shape[1]:
            return False
        return _white(frame[y, x:x + CARD_W]).mean() > 0.9 and _tan(frame[y - 2:y, x:x + CARD_W]).mean() > 0.7

    def read_corner(self, frame, x, y, *, visible_columns=RANK_W, covered=True, pips=True) -> Sighting:
        """Identify the card whose interior top-left is (x, y)."""
        notes = []
        key = phase(x, y)
        rank_region = frame[y:y + STRIP_H, x:x + RANK_W]
        rank_mask = card_ink(rank_region)
        columns = None if visible_columns >= RANK_W else visible_columns
        try:
            rank, rank_score, method = self.pack.ranks.classify(rank_mask, key, columns=columns, learn=self.learn)
        except ValueError as exc:
            raise ValueError(f"unreadable rank at ({x}, {y}): {exc}")
        colour = mean_ink_colour(rank_region, rank_mask)
        if colour is None:
            raise ValueError(f"no ink at ({x}, {y})")
        red = is_red(colour)
        suit, suit_score = None, 0.0
        if not covered or not pips:
            suit_region = frame[y + STRIP_H:y + CORNER_H, x:x + RANK_W]
            try:
                suit, suit_score, _ = self.pack.suits.classify(card_ink(suit_region), phase(x, y + STRIP_H),
                                                              columns=columns, learn=self.learn,
                                                              labels="hd" if red else "sc")
            except ValueError as exc:
                raise ValueError(f"unreadable suit glyph at ({x}, {y}): {exc}")
        else:
            # Covered tableau card: the first pip row tells the suit of a number card.
            candidates = "hd" if red else "sc"
            if 2 <= RANKS.index(rank) + 1 <= 10:
                pip_mask = card_ink(frame[y:y + STRIP_H, x + RANK_W:x + CARD_W])
                scores = {}
                for s in candidates:
                    scores[s] = max((locate(pip_mask, tight(t))[0] for t in self.pack.pips.templates[s].values()), default=0.0)
                suit = max(scores, key=scores.get)
                suit_score = scores[suit]
                other = min(scores.values())
                if suit_score < PIP_MATCH or suit_score - other < MARGIN:
                    suit, suit_score = None, 0.0
                    notes.append(f"pip row ambiguous {scores}")
            else:
                notes.append("suit hidden")
        if suit is not None and (suit in "hd") != red:
            raise ValueError(f"suit {suit} contradicts ink colour at ({x}, {y})")
        card = Card(RANKS.index(rank) + 1, suit, red)
        slot = Slot(x, y, visible_columns if visible_columns < RANK_W else CARD_W, STRIP_H if covered else CARD_H)
        return Sighting(card, slot, covered, rank_score, suit_score, method, notes)

    def white_cards(self, frame, region, *, min_height=STRIP_H - 2) -> list[tuple[int, int, int, int]]:
        """Card-like white components inside a region, as absolute bboxes sorted by x."""
        import cv2
        rx, ry, rw, rh = region
        mask = _white(frame[ry:ry + rh, rx:rx + rw]).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if h >= min_height and w >= 8 and area >= 0.2 * w * h:
                boxes.append((rx + x, ry + y, w, h))
        return sorted(boxes)

    # -- the board ------------------------------------------------------------------------------
    def read(self, frame: np.ndarray) -> Reading:
        problems = []
        columns = []
        for i, x in enumerate(COLUMN_X):
            cards, y = [], COLUMN_Y
            while self.strip_at(frame, x, y):
                covered = self.strip_at(frame, x, y + PITCH_Y)
                try:
                    cards.append(self.read_corner(frame, x, y, covered=covered))
                except ValueError as exc:
                    problems.append(f"c{i}[{len(cards)}]: {exc}")
                    break
                y += PITCH_Y
            columns.append(cards)
        waste = []
        boxes = self.white_cards(frame, WASTE_BAND, min_height=CARD_H - 4)
        for k, (x, y, w, h) in enumerate(boxes):
            covered = k < len(boxes) - 1
            try:
                s = self.read_corner(frame, x, y, visible_columns=min(w, RANK_W) if covered else RANK_W,
                                     covered=covered, pips=False)
                s.slot = Slot(x, y, w, h)
                waste.append(s)
            except ValueError as exc:
                problems.append(f"waste[{k}]: {exc}")
                break
        foundations = []
        for slot in FOUNDATIONS:
            boxes = self.white_cards(frame, slot, min_height=CARD_H - 4)
            if not boxes:
                foundations.append(None)
                continue
            x, y, w, h = boxes[-1]
            try:
                s = self.read_corner(frame, x, y, covered=False)
                s.slot = Slot(x, y, w, h)
                foundations.append(s)
            except ValueError as exc:
                problems.append(f"foundation {len(foundations)}: {exc}")
                foundations.append(None)
        cell = None
        boxes = self.white_cards(frame, CELL_SEARCH, min_height=CARD_H - 4)
        if boxes:
            x, y, w, h = boxes[-1]
            try:
                cell = self.read_corner(frame, x, y, covered=False)
                cell.slot = Slot(x, y, w, h)
            except ValueError as exc:
                problems.append(f"cell: {exc}")
        sx, sy, sw, sh = STOCK
        patch = frame[sy:sy + sh, sx:sx + sw].reshape(-1, 3).astype(int)
        stock_present = cell is None and float((patch[:, 2] - patch[:, 0]).mean()) > 25
        reading = Reading(columns, waste, foundations, stock_present, cell, problems)
        self._check(reading)
        return reading

    def _check(self, reading: Reading):
        seen = {}
        groups = [("tableau", [s for c in reading.columns for s in c]), ("waste", reading.waste),
                  ("cell", [reading.cell] if reading.cell else []),
                  ("foundation", [s for s in reading.foundations if s])]
        for where, sightings in groups:
            for s in sightings:
                if s.card.known:
                    key = str(s.card)
                    if key in seen:
                        reading.problems.append(f"duplicate {key} in {where} and {seen[key]}")
                    seen[key] = where
        if reading.visible > 52:
            reading.problems.append(f"{reading.visible} cards visible")
