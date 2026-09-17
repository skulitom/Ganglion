"""Sawayama Solitaire rules and a bounded planner over the visible state.

Sawayama is Klondike with an all face-up tableau, a stock that is dealt once and never recycled,
any card or run allowed on an empty column, and the emptied stock slot serving as one free cell.
The planner searches deterministic moves only; a deal reveals cards that the reader must supply.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import itertools
import time

from .cards import Card, SUITS

COLUMNS = 7


@dataclass(frozen=True)
class Move:
    kind: str                    # deal, column, foundation, cell, cell_column, cell_foundation
    source: str                  # "c3" (column index), "w" (waste top), "cell", or "stock"
    index: int = 0               # first card index inside a source column (runs move together)
    target: str = ""             # "c5", "f" (foundation), "cell", or "w"

    def __str__(self):
        if self.kind == "deal":
            return "deal"
        if self.source not in ("w", "cell"):
            return f"{self.source}[{self.index}] -> {self.target}"
        return f"{self.source} -> {self.target}"


@dataclass(frozen=True)
class Board:
    columns: tuple[tuple[Card, ...], ...]
    waste: tuple[Card, ...] = ()
    stock: int = 24
    foundations: tuple[int, int, int, int] = (0, 0, 0, 0)   # top rank by suit order s h d c
    cell: Card | None = None
    deal_count: int = 1

    @classmethod
    def build(cls, columns, waste=(), stock=24, foundations=None, cell=None, deal_count=1):
        columns = tuple(tuple(c) for c in columns)
        if len(columns) != COLUMNS:
            raise ValueError("Sawayama has seven columns")
        if isinstance(foundations, dict):
            found = tuple(foundations.get(s, 0) for s in SUITS)
        else:
            found = tuple(foundations or (0, 0, 0, 0))
        return cls(columns, tuple(waste), stock, found, cell, deal_count)

    @property
    def cell_open(self) -> bool:
        return self.stock == 0

    @property
    def won(self) -> bool:
        return all(r == 13 for r in self.foundations)

    def foundation_rank(self, suit: str) -> int:
        return self.foundations[SUITS.index(suit)]

    def key(self):
        return (self.columns, self.waste, self.stock, self.foundations, self.cell)

    def to_json(self) -> dict:
        return {"columns": [[str(c) for c in col] for col in self.columns], "waste": [str(c) for c in self.waste],
                "stock": self.stock, "foundations": list(self.foundations),
                "cell": str(self.cell) if self.cell else None, "deal_count": self.deal_count}

    @classmethod
    def from_json(cls, data: dict) -> "Board":
        return cls.build([[Card.parse(c) for c in col] for col in data["columns"]],
                         [Card.parse(c) for c in data["waste"]], data["stock"], tuple(data["foundations"]),
                         Card.parse(data["cell"]) if data.get("cell") else None, data.get("deal_count", 1))

    def describe(self) -> str:
        lines = [f"stock {self.stock}  waste [{' '.join(map(str, self.waste))}]  "
                 f"foundations {dict(zip(SUITS, self.foundations))}  cell {self.cell}"]
        for i, column in enumerate(self.columns):
            lines.append(f"c{i}: {' '.join(map(str, column))}")
        return "\n".join(lines)


def run_start(column: tuple[Card, ...]) -> int:
    """Index where the movable alternating descending run ending the column begins."""
    k = len(column) - 1
    while k > 0 and column[k].can_stack_on(column[k - 1]):
        k -= 1
    return max(k, 0)


def _accepts(column: tuple[Card, ...], head: Card) -> bool:
    return not column or head.can_stack_on(column[-1])


def _foundation_ready(board: Board, card: Card) -> bool:
    return card.known and board.foundation_rank(card.suit) == card.rank - 1


def legal_moves(board: Board) -> list[Move]:
    moves: list[Move] = []
    for i, column in enumerate(board.columns):
        if not column:
            continue
        top = column[-1]
        if _foundation_ready(board, top):
            moves.append(Move("foundation", f"c{i}", len(column) - 1, "f"))
        start = run_start(column)
        for k in range(start, len(column)):
            head = column[k]
            for j, other in enumerate(board.columns):
                if j == i or not _accepts(other, head):
                    continue
                if not other and k == 0:
                    continue  # moving a whole column to another empty column changes nothing
                moves.append(Move("column", f"c{i}", k, f"c{j}"))
        if board.cell_open and board.cell is None:
            moves.append(Move("cell", f"c{i}", len(column) - 1, "cell"))
    if board.waste:
        top = board.waste[-1]
        if _foundation_ready(board, top):
            moves.append(Move("foundation", "w", len(board.waste) - 1, "f"))
        for j, other in enumerate(board.columns):
            if _accepts(other, top):
                moves.append(Move("column", "w", len(board.waste) - 1, f"c{j}"))
        if board.cell_open and board.cell is None:
            moves.append(Move("cell", "w", len(board.waste) - 1, "cell"))
    if board.cell is not None:
        if _foundation_ready(board, board.cell):
            moves.append(Move("cell_foundation", "cell", 0, "f"))
        for j, other in enumerate(board.columns):
            if _accepts(other, board.cell):
                moves.append(Move("cell_column", "cell", 0, f"c{j}"))
    if board.stock > 0:
        moves.append(Move("deal", "stock", 0, "w"))
    return moves


def exposed(board: Board):
    """(card, move-to-foundation) for every card the player or the game could send up now."""
    for move in legal_moves(board):
        if move.target == "f":
            if move.source == "w":
                yield board.waste[-1], move
            elif move.source == "cell":
                yield board.cell, move
            else:
                yield board.columns[int(move.source[1:])][move.index], move


def automatic(board: Board, card: Card) -> bool:
    """Foundation moves the game performs by itself.

    Observed on 2026-09-17: exposed aces and twos go up immediately; a three stayed on the waste
    while the opposite-colour twos were still down. Higher ranks are assumed to follow the usual
    safe rule (both opposite-colour foundations already hold rank - 1); mismatches will show.
    """
    if card.rank <= 2:
        return True
    opposite = [board.foundations[i] for i, suit in enumerate(SUITS) if (suit in "hd") != card.red]
    return min(opposite) >= card.rank - 1


def _name_exposed_aces(board: Board) -> Board:
    """The game knows every suit; an exposed ace whose suit the reader could not see still goes
    up. Give it the first empty foundation of its colour so the model can follow (the next
    reading confirms or corrects the guess)."""
    board = deduce(board)

    def name(card):
        if card.rank != 1 or card.known:
            return card
        for i, suit in enumerate(SUITS):
            if (suit in "hd") == card.red and board.foundations[i] == 0:
                return Card(1, suit)
        return card
    columns = tuple(column[:-1] + (name(column[-1]),) if column else column for column in board.columns)
    waste = board.waste[:-1] + (name(board.waste[-1]),) if board.waste else board.waste
    cell = name(board.cell) if board.cell else None
    return replace(board, columns=columns, waste=waste, cell=cell)


def autoplay(board: Board) -> Board:
    """Apply the moves the game makes on its own until none remain."""
    while True:
        board = _name_exposed_aces(board)
        for card, move in exposed(board):
            if automatic(board, card):
                board = _apply(board, move)
                break
        else:
            return board


def apply(board: Board, move: Move, dealt: tuple[Card, ...] = ()) -> Board:
    """Pure transition including the automatic follow-up the game performs afterwards."""
    return autoplay(_apply(board, move, dealt))


def _apply(board: Board, move: Move, dealt: tuple[Card, ...] = ()) -> Board:
    """A deal needs the revealed cards, in the order they were turned over."""
    columns = [list(c) for c in board.columns]
    waste = list(board.waste)
    foundations = list(board.foundations)
    cell = board.cell
    stock = board.stock
    if move.kind == "deal":
        if stock <= 0:
            raise ValueError("stock is empty")
        if not 1 <= len(dealt) <= min(board.deal_count, stock):
            raise ValueError("a deal reveals between one and deal_count cards")
        waste.extend(dealt)
        stock -= len(dealt)
        return replace(board, waste=tuple(waste), stock=stock)
    if move.source == "w":
        if not waste or move.index != len(waste) - 1:
            raise ValueError("only the top waste card is playable")
        cards = [waste.pop()]
    elif move.source == "cell":
        if cell is None:
            raise ValueError("the cell is empty")
        cards, cell = [cell], None
    else:
        i = int(move.source[1:])
        column = columns[i]
        if not column or not 0 <= move.index < len(column) or move.index < run_start(tuple(column)):
            raise ValueError("not a movable run")
        cards = column[move.index:]
        del column[move.index:]
    if move.target == "f":
        if len(cards) != 1 or not _foundation_ready(board, cards[0]):
            raise ValueError("card is not next on its foundation")
        foundations[SUITS.index(cards[0].suit)] = cards[0].rank
    elif move.target == "cell":
        if not board.cell_open or board.cell is not None or len(cards) != 1:
            raise ValueError("the cell is unavailable")
        cell = cards[0]
    else:
        j = int(move.target[1:])
        if not _accepts(tuple(columns[j]), cards[0]):
            raise ValueError("run does not fit on that column")
        columns[j].extend(cards)
    return Board(tuple(tuple(c) for c in columns), tuple(waste), stock, tuple(foundations), cell,
                 board.deal_count)


def remember(previous: Board, current: Board) -> Board:
    """Carry suits seen earlier onto cards the reader now sees only by colour.

    Column and waste prefixes never reorder in Sawayama, so a position-wise match is exact.
    """
    columns = []
    for old, new in zip(previous.columns, current.columns):
        merged = list(new)
        for k in range(min(len(old), len(new))):
            if not new[k].known and old[k].known and old[k].matches(new[k]):
                merged[k] = old[k]
        columns.append(tuple(merged))
    waste = list(current.waste)
    for k in range(min(len(previous.waste), len(current.waste))):
        if not waste[k].known and previous.waste[k].known and previous.waste[k].matches(waste[k]):
            waste[k] = previous.waste[k]
    return replace(current, columns=tuple(columns), waste=tuple(waste))


def deduce(board: Board) -> Board:
    """Name a colour-only card when exactly one suit of its colour and rank is unaccounted for."""
    everywhere = [c for column in board.columns for c in column] + list(board.waste) + ([board.cell] if board.cell else [])
    known = {(c.rank, c.suit) for c in everywhere if c.known}
    for i, suit in enumerate(SUITS):
        known.update((rank, suit) for rank in range(1, board.foundations[i] + 1))

    def name(card: Card) -> Card:
        if card.known:
            return card
        candidates = [s for s in ("hd" if card.red else "sc") if (card.rank, s) not in known]
        if len(candidates) == 1:
            known.add((card.rank, candidates[0]))
            return Card(card.rank, candidates[0])
        return card
    columns = tuple(tuple(name(c) for c in column) for column in board.columns)
    waste = tuple(name(c) for c in board.waste)
    cell = name(board.cell) if board.cell else None
    return replace(board, columns=columns, waste=waste, cell=cell)


def score(board: Board) -> float:
    """Heuristic value of a position; higher is better. Not a proof of solvability."""
    value = 1000.0 * sum(board.foundations)
    value -= 12.0 * len(board.waste)
    value += 25.0 * sum(1 for c in board.columns if not c)
    if board.cell is not None:
        value -= 15.0
    needed = {s: board.foundations[i] + 1 for i, s in enumerate(SUITS)}
    for column in board.columns:
        start = run_start(column)
        # Every card above the final run is a break that must be cleared to reach what is under it.
        value -= 4.0 * start
        for k, card in enumerate(column):
            if card.known and needed[card.suit] == card.rank and k < start:
                value -= 6.0 * (start - k) * (14 - card.rank) / 13
    for k, card in enumerate(board.waste):
        if card.known and needed[card.suit] == card.rank:
            value -= 3.0 * (len(board.waste) - 1 - k)
    return value


@dataclass
class Plan:
    moves: list[Move]
    value: float
    nodes: int
    seconds: float
    reason: str = ""


def solve(board: Board, *, node_budget: int = 400000, time_budget: float = 10.0) -> Plan:
    """Depth-first search for a winning line. Meaningful once the stock is empty and every card
    is known; unknown suits simply cannot reach a foundation, which may hide a win. Returns the
    line, or an empty plan whose reason says whether the search proved a dead end or ran out."""
    started = time.perf_counter()
    if board.won:
        return Plan([], score(board), 0, 0.0, "win")
    seen = {board.key()}
    nodes = 0

    def children(state):
        out = []
        for move in legal_moves(state):
            if move.kind == "deal":
                continue
            child = apply(state, move)
            key = child.key()
            if key in seen:
                continue
            seen.add(key)
            out.append((score(child), move, child))
        out.sort(key=lambda item: -item[0])
        return out

    stack = [(board, children(board), None)]
    path: list[Move] = []
    while stack:
        nodes += 1
        if nodes > node_budget or time.perf_counter() - started > time_budget:
            return Plan([], score(board), nodes, time.perf_counter() - started, "budget")
        state, options, _ = stack[-1]
        if not options:
            stack.pop()
            if path:
                path.pop()
            continue
        _, move, child = options.pop(0)
        path.append(move)
        if child.won:
            return Plan(list(path), score(child), nodes, time.perf_counter() - started, "win")
        stack.append((child, children(child), move))
    return Plan([], score(board), nodes, time.perf_counter() - started, "no_win")


def plan(board: Board, *, node_budget: int = 20000, time_budget: float = 1.0, depth: int = 24) -> Plan:
    """Best-first search over deterministic moves; returns the path to the best position found.

    A deal is chosen only when no deterministic sequence improves the heuristic. The path may be
    empty when the stock is exhausted and nothing improves: that is a stuck position.
    """
    started = time.perf_counter()
    base = score(board)
    counter = itertools.count()
    frontier = [(-base, next(counter), board, [])]
    seen = {board.key(): base}
    best = Plan([], base, 0, 0.0, "no_improvement")
    nodes = 0
    while frontier and nodes < node_budget and time.perf_counter() - started < time_budget:
        _, _, state, path = heapq.heappop(frontier)
        nodes += 1
        if state.won:
            best = Plan(path, score(state), nodes, time.perf_counter() - started, "win")
            break
        if len(path) >= depth:
            continue
        for move in legal_moves(state):
            if move.kind == "deal":
                continue
            child = apply(state, move)
            key = child.key()
            value = score(child)
            if key in seen and seen[key] >= value:
                continue
            seen[key] = value
            child_path = path + [move]
            if value > best.value + 1e-9:
                best = Plan(child_path, value, nodes, time.perf_counter() - started, "improves")
            heapq.heappush(frontier, (-value, next(counter), child, child_path))
    best.nodes, best.seconds = nodes, time.perf_counter() - started
    if not best.moves:
        if board.stock > 0:
            best.moves, best.reason = [Move("deal", "stock", 0, "w")], "deal"
        else:
            best.reason = "stuck"
    return best
