"""Kwant-compatible graph facade over native compressed adjacency."""

from __future__ import annotations

from thouless import _core


NodeDoesNotExistError = _core.NodeDoesNotExistError
EdgeDoesNotExistError = _core.EdgeDoesNotExistError
DisabledFeatureError = _core.DisabledFeatureError


class Graph:
    """Mutable directed graph used to construct a compressed graph."""

    def __init__(self, allow_negative_nodes=False):
        self._core = _core._GraphBuilder(bool(allow_negative_nodes))

    @property
    def allow_negative_nodes(self):
        return self._core.allow_negative_nodes

    @property
    def num_nodes(self):
        return self._core.num_nodes

    @num_nodes.setter
    def num_nodes(self, value):
        self._core.num_nodes = int(value)

    def reserve(self, capacity):
        self._core.reserve(int(capacity))

    def add_edge(self, tail, head):
        return self._core.add_edge(int(tail), int(head))

    def add_edges(self, edges):
        normalized = [(int(edge[0]), int(edge[1])) for edge in edges]
        return self._core.add_edges(normalized)

    def compressed(
        self,
        twoway=False,
        edge_nr_translation=False,
        allow_lost_edges=False,
    ):
        options = (
            bool(twoway),
            bool(edge_nr_translation),
            bool(allow_lost_edges),
        )
        native = self._core.compressed(*options)
        state = (
            self.allow_negative_nodes,
            self.num_nodes,
            tuple(self._core.edges()),
            *options,
        )
        return CGraph(native, state)

    def write_dot(self, file):
        file.write(self._core.dot())


class CGraph:
    """Immutable graph with native compressed-row adjacency indices."""

    def __init__(self, native, state):
        self._core = native
        self._state = state

    @property
    def twoway(self):
        return self._core.twoway

    @property
    def edge_nr_translation(self):
        return self._core.edge_nr_translation

    @property
    def num_nodes(self):
        return self._core.num_nodes

    @property
    def num_edges(self):
        return self._core.num_edges

    @property
    def num_px_edges(self):
        return self._core.num_px_edges

    @property
    def num_xp_edges(self):
        return self._core.num_xp_edges

    def __iter__(self):
        return iter(self._core.edges())

    def has_dangling_edges(self):
        return self._core.has_dangling_edges()

    def out_neighbors(self, node):
        return self._core.out_neighbors(int(node))

    def out_edge_ids(self, node):
        return iter(self._core.out_edge_ids(int(node)))

    def in_neighbors(self, node):
        return self._core.in_neighbors(int(node))

    def in_edge_ids(self, node):
        return self._core.in_edge_ids(int(node))

    def has_edge(self, tail, head):
        return self._core.has_edge(int(tail), int(head))

    def edge_id(self, edge_number):
        return self._core.edge_id(int(edge_number))

    def first_edge_id(self, tail, head):
        return self._core.first_edge_id(int(tail), int(head))

    def all_edge_ids(self, tail, head):
        return iter(self._core.all_edge_ids(int(tail), int(head)))

    def tail(self, edge_id):
        return self._core.tail(int(edge_id))

    def head(self, edge_id):
        return self._core.head(int(edge_id))

    def write_dot(self, file):
        file.write(self._core.dot())

    def __getstate__(self):
        return self._state

    def __reduce__(self):
        return (_restore_cgraph, (self._state,))


def _restore_cgraph(state):
    (
        allow_negative_nodes,
        num_nodes,
        edges,
        twoway,
        edge_nr_translation,
        allow_lost_edges,
    ) = state
    graph = Graph(allow_negative_nodes=allow_negative_nodes)
    graph.num_nodes = num_nodes
    graph.add_edges(edges)
    return graph.compressed(
        twoway=twoway,
        edge_nr_translation=edge_nr_translation,
        allow_lost_edges=allow_lost_edges,
    )


__all__ = [
    "CGraph",
    "DisabledFeatureError",
    "EdgeDoesNotExistError",
    "Graph",
    "NodeDoesNotExistError",
]
