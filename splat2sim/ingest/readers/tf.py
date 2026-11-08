"""Tiny static-TF graph for depth->colour extrinsics."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..geometry import invert_T, make_T, quat_to_R


class TfGraph:
    def __init__(self):
        self.edges: dict[str, dict[str, np.ndarray]] = {}

    def add(self, parent: str, child: str, T_parent_child: np.ndarray) -> None:
        self.edges.setdefault(parent, {})[child] = T_parent_child
        self.edges.setdefault(child, {})[parent] = invert_T(T_parent_child)

    def add_msg(self, transform_stamped) -> None:
        tr = transform_stamped.transform
        T = make_T(quat_to_R(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w),
                   [tr.translation.x, tr.translation.y, tr.translation.z])
        self.add(transform_stamped.header.frame_id.lstrip("/"), transform_stamped.child_frame_id.lstrip("/"), T)

    def lookup(self, target: str, source: str) -> np.ndarray | None:
        """T_target_source (maps points in ``source`` into ``target``), or None if not connected."""
        target, source = target.lstrip("/"), source.lstrip("/")
        if target == source:
            return np.eye(4)
        if target not in self.edges or source not in self.edges:
            return None
        seen = {target: np.eye(4)}
        dq = deque([target])
        while dq:
            cur = dq.popleft()
            for nxt, T in self.edges[cur].items():
                if nxt not in seen:
                    seen[nxt] = seen[cur] @ T
                    if nxt == source:
                        return seen[nxt]
                    dq.append(nxt)
        return None
