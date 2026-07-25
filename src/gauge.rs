//! Gauge-independent graph cycles and oriented surface quadrature.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::Complex64;

const GAUSS_NODES: [f64; 16] = [
    0.005_299_532_504_175,
    0.027_712_488_463_384,
    0.067_184_398_806_084,
    0.122_297_795_822_499,
    0.191_061_877_798_678,
    0.270_991_611_171_386,
    0.359_198_224_610_371,
    0.452_493_745_081_181,
    0.547_506_254_918_819,
    0.640_801_775_389_629,
    0.729_008_388_828_614,
    0.808_938_122_201_322,
    0.877_702_204_177_501,
    0.932_815_601_193_916,
    0.972_287_511_536_616,
    0.994_700_467_495_825,
];
const GAUSS_WEIGHTS: [f64; 16] = [
    0.013_576_229_705_877,
    0.031_126_761_969_324,
    0.047_579_255_841_246,
    0.062_314_485_627_767,
    0.074_797_994_408_288,
    0.084_578_259_697_501,
    0.091_301_707_522_462,
    0.094_725_305_227_534,
    0.094_725_305_227_534,
    0.091_301_707_522_462,
    0.084_578_259_697_501,
    0.074_797_994_408_288,
    0.062_314_485_627_767,
    0.047_579_255_841_246,
    0.031_126_761_969_324,
    0.013_576_229_705_877,
];

/// Errors raised while constructing magnetic-gauge geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GaugeError {
    /// A point set or sampled field has an incompatible shape.
    InvalidShape,
    /// A graph edge references a node outside the declared graph.
    InvalidEdge,
    /// A coordinate or sampled field value is not finite.
    NonFiniteValue,
}

impl fmt::Display for GaugeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => write!(formatter, "gauge geometry has an invalid shape"),
            Self::InvalidEdge => write!(formatter, "gauge graph contains an invalid edge"),
            Self::NonFiniteValue => write!(formatter, "gauge geometry contains a non-finite value"),
        }
    }
}

impl std::error::Error for GaugeError {}

/// Quadrature samples for the oriented surface spanned by a polygon.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceQuadrature {
    points: Vec<Vec<f64>>,
    oriented_weights: Vec<Vec<f64>>,
}

impl SurfaceQuadrature {
    /// Cartesian quadrature points.
    #[must_use]
    pub fn points(&self) -> &[Vec<f64>] {
        &self.points
    }

    /// Oriented scalar (2D) or vector (3D) area weights.
    #[must_use]
    pub fn oriented_weights(&self) -> &[Vec<f64>] {
        &self.oriented_weights
    }

    /// Integrate scalar (2D) or vector (3D) samples over the surface.
    pub fn integrate(&self, samples: &[Vec<f64>]) -> Result<f64, GaugeError> {
        if samples.len() != self.oriented_weights.len() {
            return Err(GaugeError::InvalidShape);
        }
        self.oriented_weights
            .iter()
            .zip(samples)
            .try_fold(0.0, |integral, (weight, sample)| {
                if sample.len() != weight.len() || sample.iter().any(|value| !value.is_finite()) {
                    return Err(GaugeError::InvalidShape);
                }
                Ok(integral
                    + weight
                        .iter()
                        .zip(sample)
                        .map(|(area, field)| area * field)
                        .sum::<f64>())
            })
    }
}

/// Contract sampled fields against oriented quadrature weights.
pub fn integrate_surface_samples(
    oriented_weights: &[Vec<f64>],
    samples: &[Vec<f64>],
) -> Result<f64, GaugeError> {
    SurfaceQuadrature {
        points: vec![Vec::new(); oriented_weights.len()],
        oriented_weights: oriented_weights.to_vec(),
    }
    .integrate(samples)
}

/// Construct a fifth-order triangle quadrature for a polygonal spanning surface.
///
/// The polygon is triangulated about its vertex centroid. In two dimensions
/// each oriented weight is a scalar signed area; in three dimensions it is an
/// oriented area vector. The rule integrates polynomial fields through degree
/// five on every triangle.
pub fn surface_quadrature(loop_points: &[Vec<f64>]) -> Result<SurfaceQuadrature, GaugeError> {
    let dimension = loop_points.first().map_or(0, Vec::len);
    if loop_points.len() < 3
        || !matches!(dimension, 2 | 3)
        || loop_points.iter().any(|point| point.len() != dimension)
    {
        return Err(GaugeError::InvalidShape);
    }
    if loop_points
        .iter()
        .flatten()
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(GaugeError::NonFiniteValue);
    }
    let mut center = vec![0.0; dimension];
    for point in loop_points {
        for (coordinate, value) in center.iter_mut().zip(point) {
            *coordinate += value / loop_points.len() as f64;
        }
    }

    // Dunavant's seven-point degree-five rule, normalized so the weights sum
    // to one on each triangle.
    let barycentric = [
        ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 0.225),
        (
            [
                0.059_715_871_789_770,
                0.470_142_064_105_115,
                0.470_142_064_105_115,
            ],
            0.132_394_152_788_506,
        ),
        (
            [
                0.470_142_064_105_115,
                0.059_715_871_789_770,
                0.470_142_064_105_115,
            ],
            0.132_394_152_788_506,
        ),
        (
            [
                0.470_142_064_105_115,
                0.470_142_064_105_115,
                0.059_715_871_789_770,
            ],
            0.132_394_152_788_506,
        ),
        (
            [
                0.797_426_985_353_087,
                0.101_286_507_323_456,
                0.101_286_507_323_456,
            ],
            0.125_939_180_544_827,
        ),
        (
            [
                0.101_286_507_323_456,
                0.797_426_985_353_087,
                0.101_286_507_323_456,
            ],
            0.125_939_180_544_827,
        ),
        (
            [
                0.101_286_507_323_456,
                0.101_286_507_323_456,
                0.797_426_985_353_087,
            ],
            0.125_939_180_544_827,
        ),
    ];

    let mut points = Vec::with_capacity(7 * loop_points.len());
    let mut oriented_weights = Vec::with_capacity(7 * loop_points.len());
    for index in 0..loop_points.len() {
        let first = &loop_points[index];
        let second = &loop_points[(index + 1) % loop_points.len()];
        let oriented_area = if dimension == 2 {
            vec![
                0.5 * ((first[0] - center[0]) * (second[1] - center[1])
                    - (first[1] - center[1]) * (second[0] - center[0])),
            ]
        } else {
            let first = [
                first[0] - center[0],
                first[1] - center[1],
                first[2] - center[2],
            ];
            let second = [
                second[0] - center[0],
                second[1] - center[1],
                second[2] - center[2],
            ];
            vec![
                0.5 * (first[1] * second[2] - first[2] * second[1]),
                0.5 * (first[2] * second[0] - first[0] * second[2]),
                0.5 * (first[0] * second[1] - first[1] * second[0]),
            ]
        };
        for (coordinates, weight) in barycentric {
            points.push(
                (0..dimension)
                    .map(|axis| {
                        coordinates[0] * center[axis]
                            + coordinates[1] * first[axis]
                            + coordinates[2] * second[axis]
                    })
                    .collect(),
            );
            oriented_weights.push(oriented_area.iter().map(|area| weight * area).collect());
        }
    }
    Ok(SurfaceQuadrature {
        points,
        oriented_weights,
    })
}

fn validate_edge(first: &[f64], second: &[f64]) -> Result<usize, GaugeError> {
    let dimension = first.len();
    if !matches!(dimension, 2 | 3) || second.len() != dimension {
        return Err(GaugeError::InvalidShape);
    }
    if first
        .iter()
        .chain(second)
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(GaugeError::NonFiniteValue);
    }
    Ok(dimension)
}

/// Return the exact Poincaré-gauge phase for a spatially uniform field.
pub fn uniform_field_peierls_phase(
    first: &[f64],
    second: &[f64],
    field: &[f64],
) -> Result<Complex64, GaugeError> {
    let dimension = validate_edge(first, second)?;
    let flux = if dimension == 2 && field.len() == 1 {
        0.5 * field[0] * (first[0] * second[1] - first[1] * second[0])
    } else if dimension == 3 && field.len() == 3 {
        let cross = [
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ];
        0.5 * field
            .iter()
            .zip(cross)
            .map(|(field, area)| field * area)
            .sum::<f64>()
    } else {
        return Err(GaugeError::InvalidShape);
    };
    phase_from_flux(flux)
}

/// Convert magnetic flux to a unit-modulus Peierls phase.
pub fn phase_from_flux(flux: f64) -> Result<Complex64, GaugeError> {
    if !flux.is_finite() {
        return Err(GaugeError::NonFiniteValue);
    }
    let angle = std::f64::consts::PI * flux;
    Ok(Complex64::new(angle.cos(), angle.sin()))
}

/// Construct quadrature for a Poincaré-gauge line integral.
///
/// For a field `B`, the gauge is
/// `A(r) = ∫₀¹ s B(sr) × r ds`. A tensor Gauss rule integrates both the
/// radial homotopy and the straight hopping segment. The oriented weights are
/// scalar in 2D and vector-valued in 3D, so field callbacks remain outside the
/// native core while all geometry and contraction stay inside it.
pub fn line_phase_quadrature(
    first: &[f64],
    second: &[f64],
) -> Result<SurfaceQuadrature, GaugeError> {
    let dimension = validate_edge(first, second)?;
    let difference = first
        .iter()
        .zip(second)
        .map(|(first, second)| second - first)
        .collect::<Vec<_>>();
    let mut points = Vec::with_capacity(GAUSS_NODES.len() * GAUSS_NODES.len());
    let mut oriented_weights = Vec::with_capacity(points.capacity());
    for (segment, segment_weight) in GAUSS_NODES.into_iter().zip(GAUSS_WEIGHTS) {
        let position = first
            .iter()
            .zip(&difference)
            .map(|(first, difference)| first + segment * difference)
            .collect::<Vec<_>>();
        let orientation = if dimension == 2 {
            vec![position[0] * difference[1] - position[1] * difference[0]]
        } else {
            vec![
                position[1] * difference[2] - position[2] * difference[1],
                position[2] * difference[0] - position[0] * difference[2],
                position[0] * difference[1] - position[1] * difference[0],
            ]
        };
        for (radial, radial_weight) in GAUSS_NODES.into_iter().zip(GAUSS_WEIGHTS) {
            points.push(
                position
                    .iter()
                    .map(|coordinate| radial * coordinate)
                    .collect(),
            );
            oriented_weights.push(
                orientation
                    .iter()
                    .map(|value| segment_weight * radial_weight * radial * value)
                    .collect(),
            );
        }
    }
    Ok(SurfaceQuadrature {
        points,
        oriented_weights,
    })
}

/// Construct a 2D line-integral quadrature in a gauge adapted to an axis.
///
/// With unit vectors `t` along the axis and `n` transverse to it, this uses
/// `A_t(u, v) = -∫₀ᵛ B(u, v') dv'` and `A_n = 0`. If the field is invariant
/// along `t`, the resulting hopping phases are periodic in that direction,
/// which is required by a finalized translational lead.
pub fn axial_line_phase_quadrature(
    first: &[f64],
    second: &[f64],
    axis: &[f64],
) -> Result<SurfaceQuadrature, GaugeError> {
    if validate_edge(first, second)? != 2
        || axis.len() != 2
        || axis.iter().any(|value| !value.is_finite())
    {
        return Err(GaugeError::InvalidShape);
    }
    let norm = axis.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm == 0.0 {
        return Err(GaugeError::InvalidShape);
    }
    let tangent = [axis[0] / norm, axis[1] / norm];
    let normal = [-tangent[1], tangent[0]];
    let difference = [second[0] - first[0], second[1] - first[1]];
    let tangent_step = difference[0] * tangent[0] + difference[1] * tangent[1];
    let mut points = Vec::with_capacity(GAUSS_NODES.len() * GAUSS_NODES.len());
    let mut oriented_weights = Vec::with_capacity(points.capacity());
    for (segment, segment_weight) in GAUSS_NODES.into_iter().zip(GAUSS_WEIGHTS) {
        let position = [
            first[0] + segment * difference[0],
            first[1] + segment * difference[1],
        ];
        let longitudinal = position[0] * tangent[0] + position[1] * tangent[1];
        let transverse = position[0] * normal[0] + position[1] * normal[1];
        for (radial, radial_weight) in GAUSS_NODES.into_iter().zip(GAUSS_WEIGHTS) {
            points.push(vec![
                longitudinal * tangent[0] + radial * transverse * normal[0],
                longitudinal * tangent[1] + radial * transverse * normal[1],
            ]);
            oriented_weights.push(vec![
                -segment_weight * radial_weight * tangent_step * transverse,
            ]);
        }
    }
    Ok(SurfaceQuadrature {
        points,
        oriented_weights,
    })
}

/// Return the exact uniform-field phase in a 2D axis-adapted gauge.
pub fn uniform_axial_field_peierls_phase(
    first: &[f64],
    second: &[f64],
    axis: &[f64],
    field: f64,
) -> Result<Complex64, GaugeError> {
    if validate_edge(first, second)? != 2
        || axis.len() != 2
        || axis.iter().any(|value| !value.is_finite())
        || !field.is_finite()
    {
        return Err(GaugeError::InvalidShape);
    }
    let norm = axis.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm == 0.0 {
        return Err(GaugeError::InvalidShape);
    }
    let tangent = [axis[0] / norm, axis[1] / norm];
    let normal = [-tangent[1], tangent[0]];
    let difference = [second[0] - first[0], second[1] - first[1]];
    let tangent_step = difference[0] * tangent[0] + difference[1] * tangent[1];
    let midpoint = [0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1])];
    let transverse = midpoint[0] * normal[0] + midpoint[1] * normal[1];
    phase_from_flux(-field * tangent_step * transverse)
}

#[derive(Clone)]
struct CycleCandidate {
    nodes: Vec<usize>,
    edge_bits: Vec<u64>,
    length: usize,
}

fn bit_is_set(bits: &[u64], index: usize) -> bool {
    bits[index / 64] & (1_u64 << (index % 64)) != 0
}

fn xor_bits(left: &mut [u64], right: &[u64]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left ^= right;
    }
}

fn path_cycle(first: usize, second: usize, parent: &[usize]) -> Option<Vec<usize>> {
    let mut first_path = Vec::new();
    let mut node = first;
    loop {
        first_path.push(node);
        if parent[node] == node {
            break;
        }
        node = parent[node];
    }
    let first_positions = first_path
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index))
        .collect::<HashMap<_, _>>();
    let mut second_path = Vec::new();
    node = second;
    let (ancestor, first_index) = loop {
        if let Some(index) = first_positions.get(&node) {
            break (node, *index);
        }
        second_path.push(node);
        if parent[node] == node {
            return None;
        }
        node = parent[node];
    };
    let mut cycle = first_path[..=first_index].to_vec();
    debug_assert_eq!(cycle.last(), Some(&ancestor));
    cycle.extend(second_path.into_iter().rev());
    (cycle.len() >= 3).then_some(cycle)
}

type NormalizedGraph = (Vec<(usize, usize)>, Vec<Vec<usize>>);

fn normalized_graph(
    node_count: usize,
    undirected_edges: &[(usize, usize)],
) -> Result<NormalizedGraph, GaugeError> {
    let mut edges = undirected_edges
        .iter()
        .map(|&(first, second)| {
            if first >= node_count || second >= node_count || first == second {
                return Err(GaugeError::InvalidEdge);
            }
            Ok(if first < second {
                (first, second)
            } else {
                (second, first)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    edges.sort_unstable();
    edges.dedup();
    let mut adjacency = vec![Vec::new(); node_count];
    for &(first, second) in &edges {
        adjacency[first].push(second);
        adjacency[second].push(first);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }
    Ok((edges, adjacency))
}

type FundamentalCycleData = (Vec<(usize, usize)>, Vec<Vec<usize>>, Vec<(usize, usize)>);

fn fundamental_cycle_data(
    node_count: usize,
    undirected_edges: &[(usize, usize)],
) -> Result<FundamentalCycleData, GaugeError> {
    let (edges, adjacency) = normalized_graph(node_count, undirected_edges)?;
    let mut parent = vec![usize::MAX; node_count];
    for root in 0..node_count {
        if parent[root] != usize::MAX {
            continue;
        }
        parent[root] = root;
        let mut queue = VecDeque::from([root]);
        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency[node] {
                if parent[neighbor] == usize::MAX {
                    parent[neighbor] = node;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    let tree_edges = parent
        .iter()
        .enumerate()
        .filter_map(|(node, &parent)| {
            (node != parent).then_some(if node < parent {
                (node, parent)
            } else {
                (parent, node)
            })
        })
        .collect::<HashSet<_>>();
    let chords = edges
        .iter()
        .copied()
        .filter(|edge| !tree_edges.contains(edge))
        .collect::<Vec<_>>();
    let cycles = chords
        .iter()
        .map(|&(first, second)| path_cycle(first, second, &parent).ok_or(GaugeError::InvalidShape))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((edges, cycles, chords))
}

/// Return deterministic fundamental cycles of a breadth-first spanning forest.
///
/// Every cycle is oriented along its tree path and closes over one non-tree
/// edge. The returned order is the same order expected by
/// [`peierls_phases_from_fluxes`].
pub fn fundamental_cycles(
    node_count: usize,
    undirected_edges: &[(usize, usize)],
) -> Result<Vec<Vec<usize>>, GaugeError> {
    fundamental_cycle_data(node_count, undirected_edges).map(|(_, cycles, _)| cycles)
}

/// Assign Peierls phases to graph edges from fundamental-cycle fluxes.
///
/// Flux is measured in the convention where one unit contributes a phase of
/// `π`, matching the source tight-binding package.
pub fn peierls_phases_from_fluxes(
    node_count: usize,
    undirected_edges: &[(usize, usize)],
    cycle_fluxes: &[f64],
) -> Result<Vec<(usize, usize, Complex64)>, GaugeError> {
    let (edges, cycles, chords) = fundamental_cycle_data(node_count, undirected_edges)?;
    if cycle_fluxes.len() != cycles.len() || cycle_fluxes.iter().any(|flux| !flux.is_finite()) {
        return Err(GaugeError::InvalidShape);
    }
    let flux_by_chord = chords
        .into_iter()
        .zip(cycle_fluxes.iter().copied())
        .collect::<HashMap<_, _>>();
    Ok(edges
        .into_iter()
        .map(|(first, second)| {
            // The cycle follows the tree path from `first` to `second` and
            // closes over the chord from `second` back to `first`.
            let angle = flux_by_chord
                .get(&(first, second))
                .map_or(0.0, |flux| -std::f64::consts::PI * flux);
            (first, second, Complex64::new(angle.cos(), angle.sin()))
        })
        .collect())
}

/// Return whether an undirected graph is connected.
pub fn graph_is_connected(
    node_count: usize,
    undirected_edges: &[(usize, usize)],
) -> Result<bool, GaugeError> {
    if node_count == 0 {
        return Ok(false);
    }
    let (_, adjacency) = normalized_graph(node_count, undirected_edges)?;
    let mut visited = vec![false; node_count];
    visited[0] = true;
    let mut queue = VecDeque::from([0]);
    while let Some(node) = queue.pop_front() {
        for &neighbor in &adjacency[node] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    Ok(visited.into_iter().all(|value| value))
}

/// Return whether independent lead-interface constraints form a forest.
///
/// Each interface is a connected constraint hyperedge. A new interface closes
/// a gauge constraint loop exactly when two of its sites were already
/// connected by earlier interfaces.
pub fn interface_constraints_are_acyclic(
    node_count: usize,
    interfaces: &[Vec<usize>],
) -> Result<bool, GaugeError> {
    let mut parent = (0..node_count).collect::<Vec<_>>();

    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }

    for interface in interfaces {
        let mut nodes = interface.clone();
        nodes.sort_unstable();
        nodes.dedup();
        if nodes.iter().any(|node| *node >= node_count) {
            return Err(GaugeError::InvalidEdge);
        }
        let mut roots = HashSet::with_capacity(nodes.len());
        for &node in &nodes {
            let component = root(&mut parent, node);
            if !roots.insert(component) {
                return Ok(false);
            }
        }
        if let Some(&first) = nodes.first() {
            let first_root = root(&mut parent, first);
            for &node in &nodes[1..] {
                let node_root = root(&mut parent, node);
                parent[node_root] = first_root;
            }
        }
    }
    Ok(true)
}

/// Return a minimum cycle basis of a simple undirected graph.
///
/// Horton candidates are generated from every breadth-first shortest-path
/// tree and greedily reduced over GF(2). This yields a minimum-total-length
/// basis without assuming a planar or lattice-specific graph.
pub fn minimum_cycle_basis(
    node_count: usize,
    undirected_edges: &[(usize, usize)],
) -> Result<Vec<Vec<usize>>, GaugeError> {
    let (edges, adjacency) = normalized_graph(node_count, undirected_edges)?;
    let mut edge_index = HashMap::with_capacity(edges.len());
    for (index, &(first, second)) in edges.iter().enumerate() {
        edge_index.insert((first, second), index);
    }

    let component_count = {
        let mut visited = vec![false; node_count];
        let mut components = 0;
        for start in 0..node_count {
            if visited[start] {
                continue;
            }
            components += 1;
            visited[start] = true;
            let mut queue = VecDeque::from([start]);
            while let Some(node) = queue.pop_front() {
                for &neighbor in &adjacency[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        components
    };
    let target_rank = edges.len() + component_count - node_count;
    if target_rank == 0 {
        return Ok(Vec::new());
    }

    let words = edges.len().div_ceil(64);
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for root in 0..node_count {
        let mut parent = vec![usize::MAX; node_count];
        let mut distance = vec![usize::MAX; node_count];
        parent[root] = root;
        distance[root] = 0;
        let mut queue = VecDeque::from([root]);
        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency[node] {
                if distance[neighbor] == usize::MAX {
                    distance[neighbor] = distance[node] + 1;
                    parent[neighbor] = node;
                    queue.push_back(neighbor);
                }
            }
        }
        for &(first, second) in &edges {
            if distance[first] == usize::MAX || parent[first] == second || parent[second] == first {
                continue;
            }
            let Some(nodes) = path_cycle(first, second, &parent) else {
                continue;
            };
            let mut edge_bits = vec![0_u64; words];
            for index in 0..nodes.len() {
                let first = nodes[index];
                let second = nodes[(index + 1) % nodes.len()];
                let edge = if first < second {
                    (first, second)
                } else {
                    (second, first)
                };
                let index = edge_index[&edge];
                edge_bits[index / 64] ^= 1_u64 << (index % 64);
            }
            if seen.insert(edge_bits.clone()) {
                candidates.push(CycleCandidate {
                    length: nodes.len(),
                    nodes,
                    edge_bits,
                });
            }
        }
    }
    candidates.sort_by(|left, right| {
        left.length
            .cmp(&right.length)
            .then_with(|| left.edge_bits.cmp(&right.edge_bits))
    });

    let mut row_echelon: Vec<Option<Vec<u64>>> = vec![None; edges.len()];
    let mut basis = Vec::with_capacity(target_rank);
    for candidate in candidates {
        let mut reduced = candidate.edge_bits.clone();
        let mut independent = false;
        for pivot in (0..edges.len()).rev() {
            if !bit_is_set(&reduced, pivot) {
                continue;
            }
            if let Some(existing) = &row_echelon[pivot] {
                xor_bits(&mut reduced, existing);
            } else {
                row_echelon[pivot] = Some(reduced);
                independent = true;
                break;
            }
        }
        if independent {
            basis.push(candidate.nodes);
            if basis.len() == target_rank {
                return Ok(basis);
            }
        }
    }
    Err(GaugeError::InvalidShape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_surface_integral_has_oriented_unit_area() {
        let quadrature = surface_quadrature(&[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ])
        .unwrap();
        let samples = vec![vec![1.0]; quadrature.points().len()];
        assert!((quadrature.integrate(&samples).unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn poincare_line_quadrature_matches_uniform_field_phase() {
        let first = [0.4, -0.7];
        let second = [1.2, 0.3];
        let quadrature = line_phase_quadrature(&first, &second).unwrap();
        let flux = quadrature
            .integrate(&vec![vec![0.6]; quadrature.points().len()])
            .unwrap();
        let sampled = phase_from_flux(flux).unwrap();
        let exact = uniform_field_peierls_phase(&first, &second, &[0.6]).unwrap();
        assert!((sampled - exact).norm() < 1.0e-13);
    }

    #[test]
    fn axial_gauge_is_periodic_along_its_axis() {
        let first = [0.4, -0.7];
        let second = [1.2, 0.3];
        let shifted_first = [3.4, -0.7];
        let shifted_second = [4.2, 0.3];
        let flux = |first: &[f64], second: &[f64]| {
            let quadrature = axial_line_phase_quadrature(first, second, &[1.0, 0.0]).unwrap();
            quadrature
                .integrate(&vec![vec![0.6]; quadrature.points().len()])
                .unwrap()
        };
        assert!((flux(&first, &second) - flux(&shifted_first, &shifted_second)).abs() < 1.0e-13);
    }

    #[test]
    fn minimum_cycle_basis_prefers_four_plaquettes() {
        let mut edges = Vec::new();
        for row in 0..3 {
            for column in 0..3 {
                let node = row * 3 + column;
                if column + 1 < 3 {
                    edges.push((node, node + 1));
                }
                if row + 1 < 3 {
                    edges.push((node, node + 3));
                }
            }
        }
        let basis = minimum_cycle_basis(9, &edges).unwrap();
        assert_eq!(basis.len(), 4);
        assert!(basis.iter().all(|cycle| cycle.len() == 4));
    }

    #[test]
    fn fundamental_flux_becomes_the_oriented_loop_phase() {
        let edges = [(0, 1), (1, 2), (2, 3), (0, 3)];
        let cycles = fundamental_cycles(4, &edges).unwrap();
        assert_eq!(cycles.len(), 1);
        let phases = peierls_phases_from_fluxes(4, &edges, &[0.25]).unwrap();
        let phase_by_edge = phases
            .into_iter()
            .map(|(first, second, phase)| ((first, second), phase))
            .collect::<HashMap<_, _>>();
        let loop_phase = (0..cycles[0].len())
            .map(|index| {
                let first = cycles[0][index];
                let second = cycles[0][(index + 1) % cycles[0].len()];
                if first < second {
                    phase_by_edge[&(first, second)]
                } else {
                    phase_by_edge[&(second, first)].conj()
                }
            })
            .product::<Complex64>();
        let expected = Complex64::new(
            (0.25 * std::f64::consts::PI).cos(),
            (0.25 * std::f64::consts::PI).sin(),
        );
        assert!((loop_phase - expected).norm() < 1.0e-12);
    }

    #[test]
    fn interface_hyperedges_detect_closed_constraint_loops() {
        assert!(
            interface_constraints_are_acyclic(4, &[vec![0, 1], vec![0, 2], vec![2, 3]]).unwrap()
        );
        assert!(!interface_constraints_are_acyclic(
            4,
            &[vec![0, 1], vec![0, 2], vec![2, 3], vec![1, 3]]
        )
        .unwrap());
        assert!(!interface_constraints_are_acyclic(2, &[vec![0, 1], vec![0, 1]]).unwrap());
    }
}
