"""Pair colour and depth messages by timestamp while streaming (bounded memory)."""

from __future__ import annotations

from collections import deque
from typing import Any


class StreamPairer:
    """Feed ``push('color'|'depth', ts_ns, payload)`` in log order; returns matched ``(ts_ns, color, depth)``.

    A message is matched with the nearest message of the other kind if within ``tol_ns``. Unmatched messages
    that fall behind a match, or overflow ``max_pending``, are dropped and counted.
    """

    def __init__(self, tol_ns: int = 15_000_000, max_pending: int = 6):
        self.tol_ns = tol_ns
        self.max_pending = max_pending
        self.q: dict[str, deque] = {"color": deque(), "depth": deque()}
        self.dropped = {"color": 0, "depth": 0}

    def push(self, kind: str, ts_ns: int, payload: Any) -> list[tuple[int, Any, Any]]:
        other = "depth" if kind == "color" else "color"
        best, best_i = None, -1
        for i, (t, _) in enumerate(self.q[other]):
            dt = abs(t - ts_ns)
            if dt <= self.tol_ns and (best is None or dt < best):
                best, best_i = dt, i
        out: list[tuple[int, Any, Any]] = []
        if best_i >= 0:
            # anything older than the match in the other queue will never match again
            for _ in range(best_i):
                self.q[other].popleft()
                self.dropped[other] += 1
            t_o, p_o = self.q[other].popleft()
            color, depth = (payload, p_o) if kind == "color" else (p_o, payload)
            out.append((ts_ns if kind == "color" else t_o, color, depth))  # colour timestamp wins
        else:
            self.q[kind].append((ts_ns, payload))
            while len(self.q[kind]) > self.max_pending:
                self.q[kind].popleft()
                self.dropped[kind] += 1
        return out

    def flush(self) -> None:
        for k in self.q:
            self.dropped[k] += len(self.q[k])
            self.q[k].clear()
