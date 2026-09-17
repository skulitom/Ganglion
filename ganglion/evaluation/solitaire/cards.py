"""Playing-card values. A card whose suit is not yet visible keeps only its colour."""
from __future__ import annotations

from dataclasses import dataclass

RANKS = "A23456789TJQK"
SUITS = "shdc"
RED = frozenset("hd")


@dataclass(frozen=True, order=True)
class Card:
    rank: int                 # 1 (ace) .. 13 (king)
    suit: str | None = None   # s h d c, or None when only the colour is known
    red: bool = False

    def __post_init__(self):
        if not 1 <= self.rank <= 13:
            raise ValueError("rank must be 1..13")
        if self.suit is not None:
            if self.suit not in SUITS:
                raise ValueError("suit must be one of s h d c")
            object.__setattr__(self, "red", self.suit in RED)

    @classmethod
    def parse(cls, text: str) -> "Card":
        """'9c', 'Th', 'Qs'; '9r' / 'Kb' for a red/black card of unknown suit."""
        text = text.strip()
        if len(text) != 2 or text[0] not in RANKS or text[1] not in SUITS + "rb":
            raise ValueError(f"unrecognised card {text!r}")
        rank = RANKS.index(text[0]) + 1
        if text[1] in "rb":
            return cls(rank, None, text[1] == "r")
        return cls(rank, text[1])

    @property
    def known(self) -> bool:
        return self.suit is not None

    def __str__(self):
        return RANKS[self.rank - 1] + (self.suit or ("r" if self.red else "b"))

    def can_stack_on(self, other: "Card") -> bool:
        """Tableau rule: one rank lower and the opposite colour."""
        return self.rank + 1 == other.rank and self.red != other.red

    def matches(self, other: "Card") -> bool:
        """The same card, allowing an unknown suit on either side."""
        if self.rank != other.rank or self.red != other.red:
            return False
        return self.suit is None or other.suit is None or self.suit == other.suit


def parse_cards(text: str) -> list[Card]:
    return [Card.parse(t) for t in text.split()] if text.strip() else []
