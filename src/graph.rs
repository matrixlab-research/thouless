//! Directed graph construction and immutable compressed adjacency storage.
//!
//! The native API separates mutation from querying: [`DirectedGraphBuilder`]
//! accepts edges in arbitrary order, while [`CompressedGraph`] stores outgoing
//! and, optionally, incoming adjacency in compressed-row form.

use std::fmt;
use std::ops::Range;

/// Integer identifier for a graph node.
pub type NodeId = i64;

/// Integer identifier for an edge in a compressed graph.
pub type EdgeId = usize;

/// A directed edge. Negative endpoints represent dangling external nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectedEdge {
    tail: NodeId,
    head: NodeId,
}

impl DirectedEdge {
    /// Construct an edge from `tail` to `head`.
    ///
    /// A negative endpoint denotes a dangling external node and is accepted
    /// only by a builder created with
    /// [`DirectedGraphBuilder::allowing_dangling_nodes`].
    #[must_use]
    pub const fn new(tail: NodeId, head: NodeId) -> Self {
        Self { tail, head }
    }

    /// Return the source-node identifier.
    #[must_use]
    pub const fn tail(self) -> NodeId {
        self.tail
    }

    /// Return the destination-node identifier.
    #[must_use]
    pub const fn head(self) -> NodeId {
        self.head
    }
}

/// Options controlling which reverse indices are retained during compression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompressionOptions {
    /// Retain incoming adjacency and edges whose tail is dangling.
    pub reverse_index: bool,
    /// Retain the mapping from insertion-order edge numbers to compressed IDs.
    pub edge_number_map: bool,
    /// Permit one-way compression to discard edges whose tail is dangling.
    pub allow_discarded_edges: bool,
}

/// Errors raised by graph construction and queries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// A negative endpoint was added to a builder that forbids dangling nodes.
    NegativeNodesDisabled,
    /// Both endpoints of one edge were negative.
    DoublyDanglingEdge,
    /// An explicit node-count update attempted to remove existing nodes.
    NodeCountCannotDecrease {
        /// Node count before the rejected update.
        current: usize,
        /// Requested smaller node count.
        requested: usize,
    },
    /// Compression would discard a dangling-tail edge without permission.
    ReverseIndexRequiredForDanglingTail,
    /// A query referenced a node outside the compressed graph.
    NodeDoesNotExist(NodeId),
    /// A query referenced an absent directed edge.
    EdgeDoesNotExist,
    /// A query needs an index that was not retained during compression.
    FeatureDisabled(&'static str),
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeNodesDisabled => {
                write!(formatter, "negative node identifiers are not enabled")
            }
            Self::DoublyDanglingEdge => {
                write!(formatter, "an edge cannot have two dangling endpoints")
            }
            Self::NodeCountCannotDecrease { current, requested } => write!(
                formatter,
                "node count cannot decrease from {current} to {requested}"
            ),
            Self::ReverseIndexRequiredForDanglingTail => write!(
                formatter,
                "edges with dangling tails require a reverse index or explicit discarding"
            ),
            Self::NodeDoesNotExist(node) => {
                write!(formatter, "node {node} does not exist")
            }
            Self::EdgeDoesNotExist => write!(formatter, "edge does not exist"),
            Self::FeatureDisabled(feature) => {
                write!(formatter, "graph feature is disabled: {feature}")
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// Mutable directed graph used to assemble a [`CompressedGraph`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectedGraphBuilder {
    allow_dangling_nodes: bool,
    node_count: usize,
    edges: Vec<DirectedEdge>,
}

impl Default for DirectedGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectedGraphBuilder {
    /// Construct a graph that accepts only non-negative node identifiers.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            allow_dangling_nodes: false,
            node_count: 0,
            edges: Vec::new(),
        }
    }

    /// Construct a graph that permits one dangling endpoint per edge.
    #[must_use]
    pub const fn allowing_dangling_nodes() -> Self {
        Self {
            allow_dangling_nodes: true,
            node_count: 0,
            edges: Vec::new(),
        }
    }

    /// Return whether negative identifiers are accepted as dangling endpoints.
    #[must_use]
    pub const fn allows_dangling_nodes(&self) -> bool {
        self.allow_dangling_nodes
    }

    /// Return the number of non-dangling nodes, including isolated nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Increase the number of nodes, including isolated nodes.
    pub fn set_node_count(&mut self, node_count: usize) -> Result<(), GraphError> {
        if node_count < self.node_count {
            return Err(GraphError::NodeCountCannotDecrease {
                current: self.node_count,
                requested: node_count,
            });
        }
        self.node_count = node_count;
        Ok(())
    }

    /// Return the number of edges currently held by the mutable builder.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return edges in their insertion order.
    #[must_use]
    pub fn edges(&self) -> &[DirectedEdge] {
        &self.edges
    }

    /// Reserve storage for at least `additional` further edges.
    pub fn reserve_edges(&mut self, additional: usize) {
        self.edges.reserve(additional);
    }

    /// Add an edge and return its insertion-order edge number.
    pub fn add_edge(&mut self, edge: DirectedEdge) -> Result<usize, GraphError> {
        let tail_dangling = edge.tail < 0;
        let head_dangling = edge.head < 0;
        if tail_dangling || head_dangling {
            if !self.allow_dangling_nodes {
                return Err(GraphError::NegativeNodesDisabled);
            }
            if tail_dangling && head_dangling {
                return Err(GraphError::DoublyDanglingEdge);
            }
        }

        for node in [edge.tail, edge.head] {
            if node >= 0 {
                let required = usize::try_from(node)
                    .expect("a non-negative i64 is representable as usize on supported targets")
                    .checked_add(1)
                    .ok_or(GraphError::NodeDoesNotExist(node))?;
                self.node_count = self.node_count.max(required);
            }
        }

        let edge_number = self.edges.len();
        self.edges.push(edge);
        Ok(edge_number)
    }

    /// Add all edges, returning the insertion-order number of the first edge.
    pub fn extend_edges<I>(&mut self, edges: I) -> Result<usize, GraphError>
    where
        I: IntoIterator<Item = DirectedEdge>,
    {
        let first = self.edges.len();
        for edge in edges {
            self.add_edge(edge)?;
        }
        Ok(first)
    }

    /// Build immutable compressed adjacency indices.
    pub fn compress(&self, options: CompressionOptions) -> Result<CompressedGraph, GraphError> {
        let positive_to_positive = self
            .edges
            .iter()
            .filter(|edge| edge.tail >= 0 && edge.head >= 0)
            .count();
        let positive_to_dangling = self
            .edges
            .iter()
            .filter(|edge| edge.tail >= 0 && edge.head < 0)
            .count();
        let dangling_to_positive = self
            .edges
            .iter()
            .filter(|edge| edge.tail < 0 && edge.head >= 0)
            .count();

        if dangling_to_positive > 0 && !options.reverse_index && !options.allow_discarded_edges {
            return Err(GraphError::ReverseIndexRequiredForDanglingTail);
        }

        let outgoing_edge_count = positive_to_positive + positive_to_dangling;
        let incoming_edge_count = if options.reverse_index {
            positive_to_positive + dangling_to_positive
        } else {
            positive_to_positive
        };
        let compressed_edge_count = if options.reverse_index {
            self.edges.len()
        } else {
            outgoing_edge_count
        };

        let mut outgoing_offsets = vec![0_usize; self.node_count + 1];
        for edge in &self.edges {
            if edge.tail >= 0 {
                let tail =
                    usize::try_from(edge.tail).expect("validated non-negative node identifier");
                outgoing_offsets[tail + 1] += 1;
            }
        }
        prefix_sum(&mut outgoing_offsets);

        let mut heads = vec![0_i64; compressed_edge_count];
        let mut next_outgoing = outgoing_offsets[..self.node_count].to_vec();
        let mut next_dangling = outgoing_edge_count;
        let mut source_edge_ids = vec![None; self.edges.len()];

        for (edge_number, edge) in self.edges.iter().enumerate() {
            let edge_id = if edge.tail >= 0 {
                let tail =
                    usize::try_from(edge.tail).expect("validated non-negative node identifier");
                let edge_id = next_outgoing[tail];
                next_outgoing[tail] += 1;
                heads[edge_id] = edge.head;
                Some(edge_id)
            } else if options.reverse_index {
                let edge_id = next_dangling;
                next_dangling += 1;
                heads[edge_id] = edge.head;
                Some(edge_id)
            } else {
                None
            };
            source_edge_ids[edge_number] = edge_id;
        }

        let (incoming_offsets, tails, incoming_edge_ids) = if options.reverse_index {
            let mut offsets = vec![0_usize; self.node_count + 1];
            for edge in &self.edges {
                if edge.head >= 0 {
                    let head =
                        usize::try_from(edge.head).expect("validated non-negative node identifier");
                    offsets[head + 1] += 1;
                }
            }
            prefix_sum(&mut offsets);
            let mut tails = vec![0_i64; incoming_edge_count];
            let mut edge_ids = vec![0_usize; incoming_edge_count];
            let mut next_incoming = offsets[..self.node_count].to_vec();
            for (edge_number, edge) in self.edges.iter().enumerate() {
                if edge.head < 0 {
                    continue;
                }
                let head =
                    usize::try_from(edge.head).expect("validated non-negative node identifier");
                let incoming_index = next_incoming[head];
                next_incoming[head] += 1;
                tails[incoming_index] = edge.tail;
                edge_ids[incoming_index] = source_edge_ids[edge_number]
                    .expect("reverse-index compression retains every edge");
            }
            (Some(offsets), Some(tails), Some(edge_ids))
        } else {
            (None, None, None)
        };
        let edge_ids_by_number = options.edge_number_map.then_some(source_edge_ids);

        Ok(CompressedGraph {
            reverse_index: options.reverse_index,
            edge_number_map: options.edge_number_map,
            node_count: self.node_count,
            outgoing_edge_count,
            incoming_edge_count,
            compressed_edge_count,
            outgoing_offsets,
            heads,
            incoming_offsets,
            tails,
            incoming_edge_ids,
            edge_ids_by_number,
        })
    }
}

fn prefix_sum(offsets: &mut [usize]) {
    for index in 1..offsets.len() {
        offsets[index] += offsets[index - 1];
    }
}

/// Immutable directed graph with compressed outgoing adjacency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedGraph {
    reverse_index: bool,
    edge_number_map: bool,
    node_count: usize,
    outgoing_edge_count: usize,
    incoming_edge_count: usize,
    compressed_edge_count: usize,
    outgoing_offsets: Vec<usize>,
    heads: Vec<NodeId>,
    incoming_offsets: Option<Vec<usize>>,
    tails: Option<Vec<NodeId>>,
    incoming_edge_ids: Option<Vec<EdgeId>>,
    edge_ids_by_number: Option<Vec<Option<EdgeId>>>,
}

impl CompressedGraph {
    /// Return whether incoming adjacency was retained during compression.
    #[must_use]
    pub const fn has_reverse_index(&self) -> bool {
        self.reverse_index
    }

    /// Return whether insertion-order edge numbers can be mapped to compressed IDs.
    #[must_use]
    pub const fn has_edge_number_map(&self) -> bool {
        self.edge_number_map
    }

    /// Return the number of non-dangling nodes, including isolated nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Return the number of edges retained in compressed storage.
    ///
    /// This can be smaller than the builder edge count when dangling-tail
    /// edges were explicitly allowed to be discarded.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.compressed_edge_count
    }

    /// Edges with a non-dangling tail.
    #[must_use]
    pub const fn outgoing_edge_count(&self) -> usize {
        self.outgoing_edge_count
    }

    /// Edges with a non-dangling head.
    #[must_use]
    pub const fn incoming_edge_count(&self) -> usize {
        self.incoming_edge_count
    }

    /// Return whether any retained edge has one dangling endpoint.
    #[must_use]
    pub const fn has_dangling_edges(&self) -> bool {
        self.compressed_edge_count != self.outgoing_edge_count
            || self.compressed_edge_count != self.incoming_edge_count
    }

    fn node_index(&self, node: NodeId) -> Result<usize, GraphError> {
        if node < 0 {
            return Err(GraphError::NodeDoesNotExist(node));
        }
        let index = usize::try_from(node).map_err(|_| GraphError::NodeDoesNotExist(node))?;
        if index >= self.node_count {
            return Err(GraphError::NodeDoesNotExist(node));
        }
        Ok(index)
    }

    fn validate_pair_nodes(&self, tail: NodeId, head: NodeId) -> Result<(), GraphError> {
        for node in [tail, head] {
            if node >= 0 {
                let index =
                    usize::try_from(node).map_err(|_| GraphError::NodeDoesNotExist(node))?;
                if index >= self.node_count {
                    return Err(GraphError::NodeDoesNotExist(node));
                }
            }
        }
        Ok(())
    }

    /// Outgoing neighbors in source insertion order within a node.
    pub fn outgoing_neighbors(&self, node: NodeId) -> Result<&[NodeId], GraphError> {
        let node = self.node_index(node)?;
        Ok(&self.heads[self.outgoing_offsets[node]..self.outgoing_offsets[node + 1]])
    }

    /// Compressed IDs of outgoing edges.
    pub fn outgoing_edge_ids(&self, node: NodeId) -> Result<Range<EdgeId>, GraphError> {
        let node = self.node_index(node)?;
        Ok(self.outgoing_offsets[node]..self.outgoing_offsets[node + 1])
    }

    /// Incoming neighbors in source insertion order within a node.
    pub fn incoming_neighbors(&self, node: NodeId) -> Result<&[NodeId], GraphError> {
        if !self.reverse_index {
            return Err(GraphError::FeatureDisabled("reverse index"));
        }
        let node = self.node_index(node)?;
        let offsets = self
            .incoming_offsets
            .as_ref()
            .expect("reverse-index graph has incoming offsets");
        let tails = self
            .tails
            .as_ref()
            .expect("reverse-index graph has incoming tails");
        Ok(&tails[offsets[node]..offsets[node + 1]])
    }

    /// Compressed IDs of incoming edges.
    pub fn incoming_edge_ids(&self, node: NodeId) -> Result<&[EdgeId], GraphError> {
        if !self.reverse_index {
            return Err(GraphError::FeatureDisabled("reverse index"));
        }
        let node = self.node_index(node)?;
        let offsets = self
            .incoming_offsets
            .as_ref()
            .expect("reverse-index graph has incoming offsets");
        let ids = self
            .incoming_edge_ids
            .as_ref()
            .expect("reverse-index graph has incoming edge IDs");
        Ok(&ids[offsets[node]..offsets[node + 1]])
    }

    /// Whether at least one edge joins `tail` to `head`.
    pub fn contains_edge(&self, tail: NodeId, head: NodeId) -> Result<bool, GraphError> {
        self.validate_pair_nodes(tail, head)?;
        if tail >= 0 {
            return Ok(self.outgoing_neighbors(tail)?.contains(&head));
        }
        if !self.reverse_index {
            return Err(GraphError::FeatureDisabled("reverse index"));
        }
        if head < 0 {
            return Err(GraphError::EdgeDoesNotExist);
        }
        Ok(self.incoming_neighbors(head)?.contains(&tail))
    }

    /// Translate an insertion-order edge number into its compressed ID.
    pub fn edge_id_from_number(&self, edge_number: usize) -> Result<EdgeId, GraphError> {
        let mapping = self
            .edge_ids_by_number
            .as_ref()
            .ok_or(GraphError::FeatureDisabled("edge-number map"))?;
        mapping
            .get(edge_number)
            .copied()
            .flatten()
            .ok_or(GraphError::EdgeDoesNotExist)
    }

    /// First compressed edge ID joining `tail` to `head`.
    pub fn first_edge_id(&self, tail: NodeId, head: NodeId) -> Result<EdgeId, GraphError> {
        self.all_edge_ids(tail, head)?
            .into_iter()
            .next()
            .ok_or(GraphError::EdgeDoesNotExist)
    }

    /// All compressed edge IDs joining `tail` to `head`.
    pub fn all_edge_ids(&self, tail: NodeId, head: NodeId) -> Result<Vec<EdgeId>, GraphError> {
        self.validate_pair_nodes(tail, head)?;
        if tail >= 0 {
            return Ok(self
                .outgoing_edge_ids(tail)?
                .filter(|edge_id| self.heads[*edge_id] == head)
                .collect());
        }
        if !self.reverse_index {
            return Err(GraphError::FeatureDisabled("reverse index"));
        }
        if head < 0 {
            return Err(GraphError::EdgeDoesNotExist);
        }
        let head_index = self.node_index(head)?;
        let offsets = self
            .incoming_offsets
            .as_ref()
            .expect("reverse-index graph has incoming offsets");
        let tails = self
            .tails
            .as_ref()
            .expect("reverse-index graph has incoming tails");
        let ids = self
            .incoming_edge_ids
            .as_ref()
            .expect("reverse-index graph has incoming edge IDs");
        Ok((offsets[head_index]..offsets[head_index + 1])
            .filter_map(|index| (tails[index] == tail).then_some(ids[index]))
            .collect())
    }

    /// Tail of a compressed edge, or `None` for a dangling tail.
    pub fn tail(&self, edge_id: EdgeId) -> Result<Option<NodeId>, GraphError> {
        if edge_id >= self.compressed_edge_count {
            return Err(GraphError::EdgeDoesNotExist);
        }
        if edge_id >= self.outgoing_edge_count {
            return Ok(None);
        }
        let upper = self
            .outgoing_offsets
            .partition_point(|offset| *offset <= edge_id);
        let tail = upper.checked_sub(1).ok_or(GraphError::EdgeDoesNotExist)?;
        Ok(Some(
            NodeId::try_from(tail).expect("node count is representable as i64"),
        ))
    }

    /// Head of a compressed edge, including a dangling negative head.
    pub fn head(&self, edge_id: EdgeId) -> Result<NodeId, GraphError> {
        self.heads
            .get(edge_id)
            .copied()
            .ok_or(GraphError::EdgeDoesNotExist)
    }

    /// Iterate over all edges whose tail is non-dangling.
    pub fn edges(&self) -> impl Iterator<Item = DirectedEdge> + '_ {
        (0..self.node_count).flat_map(move |tail| {
            let range = self.outgoing_offsets[tail]..self.outgoing_offsets[tail + 1];
            range.map(move |edge_id| {
                DirectedEdge::new(
                    NodeId::try_from(tail).expect("node count is representable as i64"),
                    self.heads[edge_id],
                )
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_preserves_parallel_edges_and_adjacency_order() {
        let mut builder = DirectedGraphBuilder::new();
        for edge in [(1, 2), (0, 2), (0, 1), (1, 2)] {
            builder.add_edge(DirectedEdge::new(edge.0, edge.1)).unwrap();
        }
        let graph = builder
            .compress(CompressionOptions {
                reverse_index: true,
                edge_number_map: true,
                allow_discarded_edges: false,
            })
            .unwrap();

        assert_eq!(graph.outgoing_neighbors(0).unwrap(), [2, 1]);
        assert_eq!(graph.incoming_neighbors(2).unwrap(), [1, 0, 1]);
        assert_eq!(graph.all_edge_ids(1, 2).unwrap().len(), 2);
        for edge_number in 0..4 {
            let edge_id = graph.edge_id_from_number(edge_number).unwrap();
            let expected = builder.edges()[edge_number];
            assert_eq!(graph.tail(edge_id).unwrap(), Some(expected.tail()));
            assert_eq!(graph.head(edge_id).unwrap(), expected.head());
        }
    }

    #[test]
    fn dangling_edges_require_explicit_storage_policy() {
        let mut builder = DirectedGraphBuilder::allowing_dangling_nodes();
        builder.add_edge(DirectedEdge::new(0, -1)).unwrap();
        builder.add_edge(DirectedEdge::new(-2, 0)).unwrap();
        assert_eq!(
            builder.compress(CompressionOptions::default()),
            Err(GraphError::ReverseIndexRequiredForDanglingTail)
        );

        let graph = builder
            .compress(CompressionOptions {
                reverse_index: true,
                edge_number_map: true,
                allow_discarded_edges: false,
            })
            .unwrap();
        assert!(graph.contains_edge(0, -1).unwrap());
        assert!(graph.contains_edge(-2, 0).unwrap());
        assert_eq!(
            graph.tail(graph.edge_id_from_number(1).unwrap()).unwrap(),
            None
        );
    }

    #[test]
    fn isolated_nodes_and_query_errors_are_explicit() {
        let mut builder = DirectedGraphBuilder::new();
        builder.set_node_count(4).unwrap();
        let graph = builder.compress(CompressionOptions::default()).unwrap();
        assert_eq!(graph.node_count(), 4);
        assert!(graph.outgoing_neighbors(3).unwrap().is_empty());
        assert_eq!(
            graph.outgoing_neighbors(4),
            Err(GraphError::NodeDoesNotExist(4))
        );
        assert_eq!(
            graph.incoming_neighbors(0),
            Err(GraphError::FeatureDisabled("reverse index"))
        );
    }
}
