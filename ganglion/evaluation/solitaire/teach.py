"""Build the Sawayama template pack from labelled frames.

Labels are what an agent read from the same frames. Templates are ink masks of rank glyphs,
corner suit glyphs and first-row pip tops, stored per rendering phase. Rebuild with:

    python -m ganglion.evaluation.solitaire.teach --sources ganglion/evaluation/solitaire/pack/sources.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cards import RANKS, parse_cards
from .layout import CARD_H, CARD_W, COLUMN_X, COLUMN_Y, FOUNDATIONS, PITCH_Y, RANK_W, STRIP_H, WASTE_BAND
from .reader import CORNER_H, PACK, Reader, card_ink as ink_mask, phase


def crops(frame, columns, waste, reader, foundations=()):
    """Yield (card, x, y, covered, where) for every labelled card in a frame; where is True for
    the tableau, otherwise the visible glyph width."""
    for i, cards in enumerate(columns):
        for k, card in enumerate(cards):
            yield card, COLUMN_X[i], COLUMN_Y + PITCH_Y * k, k < len(cards) - 1, True
    boxes = reader.white_cards(frame, WASTE_BAND, min_height=90)
    if len(boxes) != len(waste):
        raise ValueError(f"{len(boxes)} waste cards found, {len(waste)} labelled")
    for k, (card, (x, y, w, h)) in enumerate(zip(waste, boxes)):
        yield card, x, y, k < len(waste) - 1, min(w, RANK_W)
    for slot, label in zip(FOUNDATIONS, foundations):
        if not label:
            continue
        found = reader.white_cards(frame, slot, min_height=CARD_H - 4)
        if not found:
            raise ValueError(f"no card found on foundation slot {slot}")
        x, y, w, h = found[-1]
        yield parse_cards(label)[0], x, y, False, RANK_W


def build(sources: list[dict], out: Path) -> dict:
    import cv2
    out = Path(out)
    for sub in ("ranks", "suits", "pips"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    written = {"ranks": set(), "suits": set(), "pips": set()}

    def save(kind, label, key, mask, width=None):
        width = mask.shape[1] if width is None else width
        existing = list((out / kind).glob(f"{label}_{key}*.png"))
        if existing:
            stem = existing[0].stem
            old_width = int(stem.rpartition("_w")[2]) if "_w" in stem else mask.shape[1]
            if old_width >= width:
                return
            existing[0].unlink()
        suffix = "" if width >= mask.shape[1] else f"_w{width}"
        cv2.imwrite(str(out / kind / f"{label}_{key}{suffix}.png"), mask)
        written[kind].add(f"{label}_{key}{suffix}")

    class Blank:
        def white_cards(self, *args, **kwargs):
            return Reader.white_cards(self, *args, **kwargs)
    reader = Blank()
    for source in sources:
        frame = cv2.imread(str(Path(source["frame"])))
        if frame is None:
            raise FileNotFoundError(source["frame"])
        columns = [parse_cards(c) for c in source["columns"]]
        waste = parse_cards(source.get("waste", ""))
        for card, x, y, covered, where in crops(frame, columns, waste, reader, source.get("foundations", ())):
            in_tableau = where is True
            width = RANK_W if in_tableau else int(where)
            rank = RANKS[card.rank - 1]
            save("ranks", rank, phase(x, y), ink_mask(frame[y:y + STRIP_H, x:x + RANK_W]), width)
            if (not covered or not in_tableau) and card.suit:
                save("suits", card.suit, phase(x, y + STRIP_H),
                     ink_mask(frame[y + STRIP_H:y + CORNER_H, x:x + RANK_W]), width)
            if covered and in_tableau and card.suit and 2 <= card.rank <= 10:
                pip_mask = ink_mask(frame[y:y + STRIP_H, x + RANK_W:x + CARD_W])
                n, labels, stats, _ = cv2.connectedComponentsWithStats(pip_mask, connectivity=8)
                if n >= 2:
                    first = min(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_LEFT])
                    bx, by, bw, bh = (int(stats[first, j]) for j in range(4))
                    save("pips", card.suit, phase(x, y), np.where(labels[by:by + bh, bx:bx + bw] == first, 255, 0).astype(np.uint8))
    return {k: sorted(v) for k, v in written.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out", default=str(PACK))
    args = parser.parse_args()
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    print(json.dumps(build(sources["frames"], Path(args.out))))


if __name__ == "__main__":
    main()
