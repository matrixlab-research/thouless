"""Compressed directed graphs backed by the Rust core."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from . import _core
from ._binding import call


class GraphBuilder:
    """Mutable directed-graph assembly before compressed storage.

    Nonnegative node identifiers define ordinary nodes. When
    ``allow_negative_nodes`` is true, one endpoint of an edge may be negative
    to represent a dangling external connection.
    """

    def __init__(self, *, allow_negative_nodes: bool = False) -> None:
        self._native = call(_core._GraphBuilder, bool(allow_negative_nodes))

    @property
    def node_count(self) -> int:
        """Number of ordinary nodes, including explicitly reserved isolates."""
        return int(self._native.num_nodes)

    @node_count.setter
    def node_count(self, value: int) -> None:
        """Increase the ordinary-node count without adding edges."""
        self._native.num_nodes = int(value)

    def add_edge(self, tail: int, head: int) -> int:
        """Add one directed edge and return its insertion-order number."""
        return int(call(self._native.add_edge, int(tail), int(head)))

    def add_edges(self, edges: Iterable[tuple[int, int]]) -> int:
        """Add directed edges and return the first insertion-order number."""
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
        """Freeze the builder into compressed adjacency storage.

        Args:
            reverse_index: Retain incoming adjacency and dangling-tail edges.
            edge_number_map: Retain insertion-number to compressed-ID mapping.
            allow_discarded_edges: Permit one-way compression to discard edges
                whose tail is dangling.

        Returns:
            Immutable compressed graph.
        """
        return CompressedGraph(
            call(
                self._native.compressed,
                bool(reverse_index),
                bool(edge_number_map),
                bool(allow_discarded_edges),
            )
        )


class CompressedGraph:
    """Immutable directed graph with compressed outgoing adjacency.

    Instances are created by :meth:`GraphBuilder.compress`. Incoming queries
    are available only when the builder retained a reverse index.
    """

    def __init__(self, native: object) -> None:
        if not isinstance(native, _core._CompressedGraph):
            raise TypeError("CompressedGraph objects are created by GraphBuilder")
        self._native = native

    @property
    def node_count(self) -> int:
        """Number of ordinary nodes, including isolates."""
        return int(self._native.num_nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges retained by compression."""
        return int(self._native.num_edges)

    def outgoing_neighbors(self, node: int) -> np.ndarray:
        """Return destination identifiers of edges leaving ``node``."""
        return np.asarray(
            call(self._native.out_neighbors, int(node)),
            dtype=np.int64,
        )

    def incoming_neighbors(self, node: int) -> np.ndarray:
        """Return source identifiers of edges entering ``node``.

        Raises :class:`~thouless.errors.ThoulessError` if the graph was
        compressed without a reverse index.
        """
        return np.asarray(
            call(self._native.in_neighbors, int(node)),
            dtype=np.int64,
        )

    def contains_edge(self, tail: int, head: int) -> bool:
        """Return whether the exact directed edge ``tail -> head`` exists."""
        return bool(call(self._native.has_edge, int(tail), int(head)))

    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return all retained edges in compressed-ID order."""
        return tuple(
            (int(tail), int(head))
            for tail, head in call(self._native.edges)
        )


__all__ = ["CompressedGraph", "GraphBuilder"]
