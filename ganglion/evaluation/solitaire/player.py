"""Play Sawayama Solitaire live through Ganglion's public MCP tools.

The loop is: look, read the board, plan, execute one move as a taught drag (or a stock click),
then read again and check that the application did what the plan expected. The core provides
pixels and bounded cursor-feedback drags; every card rule lives here. Nothing is launched,
focused, dealt or closed by this module: attach it to a core bound to a visible game window.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from ganglion.arena.demo import dependency_versions
from .cards import Card
from .layout import CARD_H, CARD_W, CELL, COLUMN_X, COLUMN_Y, FOUNDATIONS, PITCH_Y, STOCK, STRIP_H, Slot
from .reader import Reader, Reading
from .sawayama import Board, Move, apply, deduce, plan, remember, solve

CARD_RGB = (250, 250, 245)
PARK = (1150, 690)              # a point on the desktop background, away from every card
LANDING_PIXELS = 100            # white area that only a card face can put into the landing band
DEAL_COUNT = 3


class PlayError(RuntimeError):
    pass


def boards_agree(expected: Board, actual: Board) -> bool:
    """Structural equality that tolerates suits the reader could not see."""
    if len(expected.columns) != len(actual.columns) or expected.stock != actual.stock:
        return False
    for a, b in zip(expected.columns + (expected.waste,), actual.columns + (actual.waste,)):
        if len(a) != len(b) or not all(x.matches(y) for x, y in zip(a, b)):
            return False
    if (expected.cell is None) != (actual.cell is None) or (expected.cell and not expected.cell.matches(actual.cell)):
        return False
    return expected.foundations == actual.foundations


def deal_agrees(before: Board, actual: Board, deal_count: int) -> bool:
    """A deal reveals min(deal_count, stock) cards; an exposed ace may already have gone up."""
    expected = min(deal_count, before.stock)
    revealed = (len(actual.waste) - len(before.waste)) + (sum(actual.foundations) - sum(before.foundations))
    if revealed != expected or actual.stock != before.stock - expected:
        return False
    if len(actual.waste) < len(before.waste) or not all(a.matches(b) for a, b in zip(before.waste, actual.waste)):
        return False
    return boards_agree(Board.build(before.columns, actual.waste, actual.stock, actual.foundations, before.cell, deal_count),
                        actual)


def landing(reading: Reading, move: Move) -> tuple[Slot, tuple[int, int], tuple[int, int, int, int]]:
    """Where the moved card's top-left must end up, the condition region, and its size."""
    if move.target == "cell":
        x, y, w, h = CELL
        slot = Slot(x, y, CARD_W, CARD_H)
        return slot, (x, y), (x + 8, y + 40, CARD_W - 16, 12)
    if move.target.startswith("c"):
        j = int(move.target[1:])
        column = reading.columns[j]
        if column:
            top = column[-1].slot
            slot = Slot(COLUMN_X[j], top.y + PITCH_Y, CARD_W, STRIP_H)
            # A band just under the current top card: background now, the landed face later.
            # Wide and tall enough that pips cannot cut every white segment below LANDING_PIXELS.
            region = (COLUMN_X[j] + 8, top.y + CARD_H + 4, CARD_W - 16, 12)
        else:
            slot = Slot(COLUMN_X[j], COLUMN_Y, CARD_W, STRIP_H)
            region = (COLUMN_X[j] + 8, COLUMN_Y + 3, CARD_W - 16, 12)
        return slot, (slot.x, slot.y), region
    if move.target == "f":
        card = _moved_card(reading, move)
        suits = reading.foundation_suits()
        index = suits.index(card.suit) if card.suit in suits else suits.index(None)
        x, y, w, h = FOUNDATIONS[index]
        slot = Slot(x, y + 1, CARD_W, CARD_H)
        region = (x + 8, y + 40, CARD_W - 16, 12)
        return slot, (slot.x, slot.y), region
    if move.target == "cell":
        x, y, w, h = CELL
        slot = Slot(x, y, CARD_W, CARD_H)
        return slot, (x, y), (x + 8, y + 40, CARD_W - 16, 12)
    raise ValueError(f"no landing for {move}")


def _moved_card(reading: Reading, move: Move) -> Card:
    if move.source == "w":
        return reading.waste[-1].card
    if move.source == "cell":
        return reading.cell.card
    return reading.columns[int(move.source[1:])][move.index].card


def source_slot(reading: Reading, move: Move) -> Slot:
    if move.source == "w":
        return reading.waste[-1].slot
    if move.source == "cell":
        return reading.cell.slot
    return reading.columns[int(move.source[1:])][move.index].slot


def grab_region(src: Slot, attempt: int = 0) -> tuple[int, int, int, int]:
    """A white margin of the card to grab by: the right edge of the visible face, which no rank
    glyph, pip row or picture reaches, so its largest white component keeps a stable centre even
    with the pointer sprite over the card. The second attempt widens the band leftwards."""
    height = min(30 if src.h > STRIP_H else STRIP_H - 2, src.h - 2)
    if attempt == 0:
        # Between the two pip columns, above any centre pip: the pointer then sits near the
        # middle of the card, which the game needs for a drop onto an empty column.
        return (src.x + 34, src.y + 1, 7, min(14, height))
    if attempt == 1:
        # Face cards fill the middle with a picture: grab the white around the rank glyph
        # instead, which keeps the pointer inside the slot a card is dropped on.
        return (src.x + 1, src.y + 1, 17, min(14, height))
    if attempt == 2:
        return (src.x + src.w - 7, src.y + 1, 6, height)   # right margin, clear of pictures
    return (src.x + src.w - 18, src.y + 1, 17, height)


class Session:
    def __init__(self, endpoint: str, out: Path, *, expected_session: int | None, deal_count: int,
                 learn: bool, verbose: bool, controller: str = "deterministic"):
        self.endpoint, self.out = endpoint, out
        self.controller = controller
        self.expected_session, self.deal_count = expected_session, deal_count
        self.reader = Reader(learn=learn)
        self.verbose = verbose
        self.report = {"measured_at": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
                       "dependencies": dependency_versions(), "endpoint": str(endpoint), "deal_count": deal_count,
                       "controller": controller,
                       "steps": [], "events": [], "lost_events": 0, "images": [], "passed": False}
        self.knowledge: Board | None = None
        self.cursor = None
        self.origin = (0, 0)
        self.snapshot = None
        self.session = None
        self.call = None
        self.frames = 0

    # -- transport ------------------------------------------------------------------------------
    async def _call(self, name, args=None, *, with_content=False):
        started = time.perf_counter()
        result = await self.session.call_tool("ganglion_" + name, args or {})
        if result.is_error:
            raise PlayError(f"{name}: " + " ".join(getattr(c, "text", "") for c in result.content))
        structured = result.structured_content
        if isinstance(structured, dict) and structured.get("lost_events"):
            self.report.setdefault("losses", []).append(
                {"call": name, "lost": structured["lost_events"], "seconds": round(time.perf_counter() - started, 3),
                 "step": len(self.report["steps"])})
        if with_content:
            return structured, result.content
        return structured

    async def look(self, *, image=True, save: str | None = None):
        r, content = await self._call("look", {"cursor": self.cursor, "image": image, "limit": 200,
                                               "max_width": 1280, "image_format": "png"}, with_content=True)
        self.cursor = r["next_cursor"]
        self.report["events"].extend(r["events"])
        self.report["lost_events"] += r["lost_events"]
        while r["has_more"]:
            more = await self._call("look", {"cursor": self.cursor, "image": False, "limit": 200})
            self.cursor = more["next_cursor"]
            self.report["events"].extend(more["events"])
            self.report["lost_events"] += more["lost_events"]
            r["has_more"] = more["has_more"]
        self.snapshot = r["snapshot"]
        frame = None
        if image:
            import cv2
            picture = next((c for c in content if getattr(c, "type", "") == "image"), None)
            if picture is None:
                raise PlayError("look returned no image")
            data = base64.b64decode(picture.data)
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if save:
                path = self.out / f"{save}.png"
                cv2.imwrite(str(path), frame)
                self.report["images"].append(path.name)
        return r, frame

    async def wait(self, kinds, timeout):
        r = await self._call("wait", {"cursor": self.cursor, "timeout": timeout, "kinds": kinds})
        self.cursor = r["next_cursor"]
        self.report["events"].extend(r["events"])
        self.report["lost_events"] += r["lost_events"]
        await self.drain_ledger()
        return r

    async def drain_ledger(self):
        """Page the ledger to its frontier; pointer feedback at 100 Hz overruns a lazy reader."""
        while True:
            more = await self._call("look", {"cursor": self.cursor, "image": False, "limit": 200})
            self.cursor = more["next_cursor"]
            self.report["events"].extend(more["events"])
            self.report["lost_events"] += more["lost_events"]
            if not more["has_more"]:
                return

    async def status(self):
        return await self._call("status")

    # -- perception ------------------------------------------------------------------------------
    def crop(self, frame):
        x, y, w, h = self.snapshot["target"]["rect"]
        return frame[y:y + h, x:x + w]

    async def read_stable(self, *, save: str | None = None, attempts: int = 8) -> tuple[Reading, np.ndarray]:
        """Two consecutive identical readings, with the pointer parked away from the cards."""
        previous = None
        last_problem = None
        for attempt in range(attempts):
            r, frame = await self.look(save=save if attempt == 0 else None)
            reading = self.reader.read(self.crop(frame))
            if reading.problems:
                last_problem = reading.problems
                previous = None
                await asyncio.sleep(0.25)
                continue
            text = reading.describe()
            if previous == text:
                return reading, frame
            previous = text
            await asyncio.sleep(0.15)
        raise PlayError(f"board did not settle: {last_problem or 'readings kept changing'}")

    # -- action ----------------------------------------------------------------------------------
    async def drain(self, timeout=2.0):
        """Wait until the core has no queued output; a new pointer command is refused until then."""
        until = time.perf_counter() + timeout
        while time.perf_counter() < until:
            s = await self.status()
            if not s["pending_commands"] and not (s["intent"] and s["intent"]["phase"] in ("running", "clicking")):
                return s
            await asyncio.sleep(0.03)
        raise PlayError("core output did not drain")

    async def park(self):
        await self.drain()
        await self._call("input", {"spec": {"action": "move", "snapshot_id": self.snapshot["id"],
                                            "point": [self.origin[0] + PARK[0], self.origin[1] + PARK[1]]}})
        await self.wait(["pointer_feedback", "input_cancelled", "input_failed"], 1)
        await asyncio.sleep(0.05)

    async def teach(self, name, region, min_pixels):
        x, y, w, h = region
        spec = {"name": name, "snapshot_id": self.snapshot["id"],
                "region": [self.origin[0] + x, self.origin[1] + y, w, h],
                "color_rgb": list(CARD_RGB), "tolerance": 18, "min_pixels": min_pixels}
        return (await self._call("watch", {"spec": spec}))["watch_id"]

    async def unwatch(self, watch_id):
        try:
            await self._call("unwatch", {"watch_id": watch_id})
        except PlayError:
            pass

    async def deal(self, reading: Reading, step: dict):
        x, y, w, h = STOCK
        point = [self.origin[0] + x + w // 2, self.origin[1] + y + h // 2]
        step["click"] = await self._call("input", {"spec": {"action": "click", "snapshot_id": self.snapshot["id"], "point": point}})
        await self.wait(["input_released", "input_failed", "input_cancelled"], 2)
        await asyncio.sleep(0.5)

    async def drag(self, reading: Reading, move: Move, step: dict):
        src = source_slot(reading, move)
        slot, target_xy, region = landing(reading, move)
        source_id, detection, grab_rect = None, None, None
        for attempt in range(4):
            grab_rect = grab_region(src, attempt)
            source_id = await self.teach("source", grab_rect, 25)   # the pointer sprite may cover half the band
            for _ in range(20):
                await asyncio.sleep(0.03)
                s = await self.status()
                watch = next((w for w in s["watches"] if w["id"] == source_id), None)
                if watch and watch["present"]:
                    detection = watch["detection"]
                    break
            if detection is not None:
                break
            await self.unwatch(source_id)
        if detection is None:
            raise PlayError(f"source card not detected in {grab_rect}")
        # A foundation pile already shows a white face, so its landing is not a colour condition:
        # that drop is released on cursor arrival and checked by reading the board afterwards.
        condition_id = None if move.target == "f" else await self.teach("landing", region, LANDING_PIXELS)
        step["watches"] = {"source": source_id, "landing": condition_id, "source_rect": src.rect,
                           "grab_region": grab_rect, "landing_region": region}
        try:
            # The grab point is the detected white pixel; align the card's top-left with the landing slot.
            grab = (detection["x"] - (self.origin[0] + src.x), detection["y"] - (self.origin[1] + src.y))
            destination = [self.origin[0] + target_xy[0] + grab[0], self.origin[1] + target_xy[1] + grab[1]]
            step["grab_offset"], step["destination"] = grab, destination
            spec = {"program": "drag", "watch_id": source_id, "destination": destination,
                    "timeout_seconds": 8, "verification_seconds": 1.2, "settle_ms": 60,
                    "speed_px_s": 700, "tolerance_px": 6, "controller": self.controller}
            if condition_id is not None:
                spec["until"] = {"watch_id": condition_id, "present": True}
            started = time.perf_counter()
            step["intent"] = await self._call("intent", {"spec": spec})
            outcome = None
            while time.perf_counter() - started < 10:
                await self.wait(["intent_completed", "intent_failed", "intent_cancelled"], 0.5)
                s = await self.status()
                outcome = s["intent"]
                if outcome["phase"] not in ("running", "clicking"):
                    break
            if outcome["phase"] in ("running", "clicking"):
                await self._call("cancel", {"intent_id": outcome["intent_id"]})
                await self.wait(["output_halted"], 1)
                s = await self.status()
                outcome = s["intent"]
            step["outcome"] = {k: outcome.get(k) for k in ("phase", "reason", "stage", "condition_verified", "cursor", "commands",
                                                           "released_mono", "controller", "neural_commands", "overridden_commands")}
            step["drag_seconds"] = time.perf_counter() - started
        finally:
            await self.unwatch(source_id)
            if condition_id is not None:
                await self.unwatch(condition_id)
        await asyncio.sleep(0.35)

    # -- the game ----------------------------------------------------------------------------------
    async def play(self, *, max_moves: int, seconds: float, dry_run: bool):
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        params = StdioServerParameters(command=sys.executable, args=["-m", "ganglion.mcp", "--endpoint", str(self.endpoint)])
        deadline = time.perf_counter() + seconds
        rejected: dict[str, set[str]] = {}
        claimed = False
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
                try:
                    # Start reading at the ledger frontier: on a long-lived core, everything before this
                    # run is already gone from the rings and would otherwise count as lost.
                    self.cursor = (await self.status())["latest_cursor"]
                    r, frame = await self.look(save="start")
                    snapshot = self.snapshot
                    self.report["target"], self.report["session_id"] = snapshot["target"], snapshot["session_id"]
                    if self.expected_session is not None and snapshot["session_id"] != self.expected_session:
                        raise PlayError(f"core runs in Windows session {snapshot['session_id']}, expected {self.expected_session}")
                    x, y, w, h = snapshot["target"]["rect"]
                    if (w, h) != (1280, 720):
                        raise PlayError(f"expected a 1280x720 client, found {w}x{h}")
                    self.origin = (x, y)
                    if dry_run:
                        reading = self.reader.read(self.crop(frame))
                        board = reading.board(self.deal_count)
                        p = plan(board)
                        self.report["dry_run"] = {"reading": reading.describe(), "plan": [str(m) for m in p.moves],
                                                  "reason": p.reason, "value": p.value, "nodes": p.nodes}
                        print(reading.describe())
                        print("plan:", p.reason, [str(m) for m in p.moves])
                        self.report["passed"] = reading.readable
                        return self.report
                    await self._call("claim", {"seconds": 60})
                    claimed = True
                    await self.park()
                    reading, frame = await self.read_stable()
                    board = reading.board(self.deal_count)
                    if self.knowledge is not None:
                        board = remember(self.knowledge, board)
                    board = deduce(board)
                    self.report["initial_board"] = board.describe()
                    moves = 0
                    visited = {board.key()}
                    line: list[Move] = []          # a proven winning line, followed while the board agrees
                    consecutive = 0
                    while moves < max_moves and time.perf_counter() < deadline:
                        await self._call("renew", {"seconds": 60})
                        if board.won:
                            self.report["result"] = "won"
                            break
                        if line:
                            p = plan.__class__ if False else None
                            p = type("Line", (), {})()
                            p.moves, p.reason, p.nodes, p.seconds = list(line), "line", 0, 0.0
                        elif board.stock == 0 and all(c.known for column in board.columns for c in column):
                            await self._call("renew", {"seconds": 60})
                            p = solve(board, time_budget=3.0)
                            self.report.setdefault("solver", []).append(
                                {"step": moves, "reason": p.reason, "nodes": p.nodes, "seconds": round(p.seconds, 3),
                                 "length": len(p.moves)})
                            if p.reason == "win":
                                line = list(p.moves)
                            elif p.reason == "no_win":
                                self.report["result"] = "stuck"
                                break
                            else:
                                p = plan(board, time_budget=1.5)
                        else:
                            p = plan(board, time_budget=1.5)
                        key = repr(board.key())

                        def usable(m):
                            if str(m) in rejected.get(key, set()):
                                return False
                            try:
                                return m.kind == "deal" or apply(board, m).key() not in visited
                            except ValueError:
                                return False
                        candidates = [m for m in p.moves[:1] if usable(m)]
                        if not candidates:
                            # The best move was rejected here or repeats a position; try another legal move once.
                            from .sawayama import legal_moves
                            candidates = [m for m in legal_moves(board) if usable(m)]
                            candidates.sort(key=lambda m: (m.kind == "deal", str(m)))
                        if not candidates or (p.reason == "stuck" and board.stock == 0):
                            self.report["result"] = "stuck"
                            break
                        move = candidates[0]
                        if line and line[0] == move:
                            line.pop(0)
                        else:
                            line = []
                        step = {"index": moves, "move": str(move), "kind": move.kind, "plan": [str(m) for m in p.moves],
                                "plan_reason": p.reason, "plan_nodes": p.nodes, "plan_seconds": round(p.seconds, 3),
                                "board_before": board.describe()}
                        started = time.perf_counter()
                        if self.verbose:
                            print(f"[{moves}] {move}  ({p.reason}, {p.nodes} nodes)", flush=True)
                        try:
                            if move.kind == "deal":
                                await self.deal(reading, step)
                            else:
                                await self.drag(reading, move, step)
                            await self.park()
                            after, frame = await self.read_stable()
                        except PlayError as exc:
                            step["error"] = str(exc)
                            self.report["steps"].append(step)
                            await self.look(save=f"error-{moves:03d}")
                            raise
                        actual = deduce(remember(board, after.board(self.deal_count)))
                        try:
                            if move.kind == "deal":
                                ok = deal_agrees(board, actual, self.deal_count)
                            else:
                                ok = boards_agree(apply(board, move), actual)
                        except ValueError as exc:
                            step["verification_error"] = str(exc)
                            ok = False
                        step.update(board_after=actual.describe(), accepted=ok, seconds=round(time.perf_counter() - started, 3))
                        if not ok:
                            line = []
                            unchanged = boards_agree(board, actual)
                            step["unchanged"] = unchanged
                            if unchanged:
                                rejected.setdefault(key, set()).add(str(move))
                                step["note"] = "application rejected the move; it is excluded at this position"
                            else:
                                step["note"] = "board changed differently than planned; continuing from what is visible"
                            await self.look(save=f"mismatch-{moves:03d}")
                            self.report["mismatches"] = self.report.get("mismatches", 0) + 1
                            consecutive += 1
                            if consecutive >= 4:
                                self.report["result"] = "too_many_mismatches"
                                self.report["steps"].append(step)
                                break
                        else:
                            consecutive = 0
                        self.report["steps"].append(step)
                        board, reading = actual, after
                        self.report["final_cards"] = board.to_json()
                        visited.add(board.key())
                        moves += 1
                    else:
                        self.report["result"] = "move_budget" if moves >= max_moves else "time_budget"
                    self.report["final_board"] = board.describe()
                    self.report["final_cards"] = board.to_json()
                    self.report["moves"] = moves
                    self.report["won"] = board.won
                    await self.look(save="end")
                    self.report["passed"] = (self.report["result"] in ("won", "move_budget", "time_budget", "stuck")
                                             and not self.report["lost_events"]
                                             and not any(e["kind"] == "input_failed" for e in self.report["events"]))
                except Exception as exc:
                    self.report["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    if claimed:
                        try:
                            await self._call("halt")
                            await asyncio.sleep(0.15)
                            s = await self.status()
                            self.report.update(final_state=s["state"], pending_commands=s["pending_commands"],
                                               capture_sources=s["capture_sources"], dropped_frames=s["dropped_frames"])
                            self.report["passed"] &= s["state"] == "halted" and not s["pending_commands"]
                        except Exception as exc:
                            self.report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                            self.report["passed"] = False
        return self.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--out", required=True, help="New output directory; existing results are never overwritten")
    parser.add_argument("--session", type=int, help="Windows session the core must run in")
    parser.add_argument("--deal-count", type=int, default=DEAL_COUNT)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--dry-run", action="store_true", help="read and plan only; no lease, no input")
    parser.add_argument("--no-learn", action="store_true", help="do not record new glyph phases into the pack")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--knowledge", help="report.json of an earlier run on the same deal; carries suits already seen")
    parser.add_argument("--controller", choices=["deterministic", "connectome"], default="deterministic",
                        help="connectome: the core must run with --shadow-checkpoint; proposals drive drags under supervision")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    session = Session(args.endpoint, out, expected_session=args.session, deal_count=args.deal_count,
                      learn=not args.no_learn, verbose=not args.quiet, controller=args.controller)
    if args.knowledge:
        earlier = json.loads(Path(args.knowledge).read_text(encoding="utf-8"))
        if earlier.get("final_cards"):
            session.knowledge = Board.from_json(earlier["final_cards"])
    report = asyncio.run(session.play(max_moves=args.max_moves, seconds=args.seconds, dry_run=args.dry_run))
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("events", "steps", "dependencies")}
    print(json.dumps(summary, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
