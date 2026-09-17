"""Two bounded event rings, one ordered stream with explicit gaps and reader-owned cursors."""
from __future__ import annotations

from collections import deque
from threading import RLock
from uuid import uuid4


class Ledger:
    def __init__(self, perception_capacity=512, action_capacity=512):
        if min(perception_capacity, action_capacity) < 1:
            raise ValueError("ledger capacities must be positive")
        self.epoch = uuid4().hex
        self.seq = 0
        self.perception = deque(maxlen=perception_capacity)
        self.actions = deque(maxlen=action_capacity)
        self.lock = RLock()

    def append(self, kind, t, *, critical=False, **data):
        with self.lock:
            self.seq += 1
            event = {"seq": self.seq, "t_mono": t, "kind": kind, **data}
            (self.actions if critical else self.perception).append(event)
            return event

    def cursor(self, seq=None):
        return f"{self.epoch}:{self.seq if seq is None else seq}"

    def decode(self, cursor):
        if cursor is None:
            return 0
        try:
            epoch, seq = cursor.split(":")
            seq = int(seq)
        except (ValueError, AttributeError):
            raise ValueError("invalid cursor; use next_cursor from look") from None
        if epoch != self.epoch:
            raise ValueError("cursor belongs to another core; start a new look")
        if not 0 <= seq <= self.seq:
            raise ValueError("cursor is outside the available sequence")
        return seq

    def read(self, cursor=None, limit=40):
        with self.lock:
            after = self.decode(cursor)
            retained = sorted((*self.perception, *self.actions), key=lambda e: e["seq"])
            pending = [e for e in retained if e["seq"] > after]
            page = pending[:limit]
            more = len(pending) > len(page)
            frontier = page[-1]["seq"] if more else self.seq
            return {
                "events": list(page),
                "next_cursor": self.cursor(frontier),
                "has_more": more,
                "lost_events": frontier - after - len(page),
                "oldest_available": retained[0]["seq"] if retained else None,
                "latest_seq": self.seq,
            }
