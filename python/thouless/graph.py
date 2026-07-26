"""Compressed directed graphs backed by the Rust core."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from . import _core
from ._binding import call


class GraphBuilder:
    def __init__(self, *, allow_negative_nodes: bool = False) -> None:
        self._native = call(_core._GraphBuilder, bool(allow_negative_nodes))

    @property
    def node_count(self) -> int:
        return int(self._native.num_nodes)

    @node_count.setter
    def node_count(self, value: int) -> None:
        self._native.num_nodes = int(value)

    def add_edge(self, tail: int, head: int) -> int:
        return int(call(self._native.add_edge, int(tail), int(head)))

    def add_edges(self, edges: Iterable[tuple[int, int]]) -> int:
        return int(
            call(
                self._native.add_edges,
                [(int(tail), int(head)) for tail, head in edges],
            )
        )

    def compress(
        self,
        *,
        reverse_index: bool = False,
        edge_number_map: bool = False,
        allow_discarded_edges: bool = False,
    ) -> "CompressedGraph":
        return CompressedGraph(
            call(
                self._native.compressed,
                bool(reverse_index),
                bool(edge_number_map),
                bool(allow_discarded_edges),
            )
        )


class CompressedGraph:
    def __init__(self, native: object) -> None:
        if not isinstance(native, _core._CompressedGraph):
            raise TypeError("CompressedGraph objects are created by GraphBuilder")
        self._native = native

    @property
    def node_count(self) -> int:
        return int(self._native.num_nodes)

    @property
    def edge_count(self) -> int:
        return int(self._native.num_edges)

    def outgoing_neighbors(self, node: int) -> np.ndarray:
        return np.asarray(
            call(self._native.out_neighbors, int(node)),
            dtype=np.int64,
        )

    def incoming_neighbors(self, node: int) -> np.ndarray:
        return np.asarray(
            call(self._native.in_neighbors, int(node)),
            dtype=np.int64,
        )

    def contains_edge(self, tail: int, head: int) -> bool:
        return bool(call(self._native.has_edge, int(tail), int(head)))

    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(tail), int(head))
            for tail, head in call(self._native.edges)
        )


__all__ = ["CompressedGraph", "GraphBuilder"]
