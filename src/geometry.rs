//! Geometry transformations and sampling in real and reciprocal space.

use std::f64::consts::TAU;

use nalgebra::DMatrix;

use crate::model::Lattice;
use crate::GeometryError;

/// A piecewise-linear reciprocal-space path sampled by Cartesian arc length.
#[derive(Clone, Debug, PartialEq)]
pub struct ReciprocalPath {
    reduced_points: Vec<Vec<f64>>,
    distances: Vec<f64>,
    node_distances: Vec<f64>,
}

impl ReciprocalPath {
    /// Samples a path through reduced-coordinate nodes.
    ///
    /// Segment lengths and sample allocation use the Cartesian reciprocal
    /// metric induced by the periodic primitive vectors. Returned momenta stay
    /// in reduced coordinates so they can be passed directly to a Bloch model.
    pub fn through(
        lattice: &Lattice,
        nodes: &[Vec<f64>],
        sample_count: usize,
    ) -> Result<Self, GeometryError> {
        let periodic_dimension = lattice.periodic_dimension();
        if periodic_dimension == 0 {
            return Err(GeometryError::NoPeriodicDirections);
        }
        if nodes.len() < 2 {
            return Err(GeometryError::InsufficientPathNodes);
        }
        if sample_count < nodes.len() {
            return Err(GeometryError::InsufficientPathSamples {
                minimum: nodes.len(),
                actual: sample_count,
            });
        }
        for (node, values) in nodes.iter().enumerate() {
            if values.len() != periodic_dimension {
                return Err(GeometryError::InvalidPathNode {
                    node,
                    expected: periodic_dimension,
                    actual: values.len(),
                });
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(GeometryError::NonFinitePathNode);
            }
        }

        let real_dimension = lattice.real_dimension();
        let periodic_vectors =
            DMatrix::from_fn(periodic_dimension, real_dimension, |periodic, cartesian| {
                lattice.primitive_vectors()[lattice.periodic_axes()[periodic]][cartesian]
            });
        let gram = &periodic_vectors * periodic_vectors.transpose();
        let reciprocal_vectors = gram
            .try_inverse()
            .ok_or(GeometryError::SingularPeriodicGeometry)?
            * periodic_vectors
            * TAU;

        let mut segment_lengths = Vec::with_capacity(nodes.len() - 1);
        for pair in nodes.windows(2) {
            let mut squared_length = 0.0;
            for cartesian in 0..real_dimension {
                let component: f64 = (0..periodic_dimension)
                    .map(|axis| {
                        (pair[1][axis] - pair[0][axis]) * reciprocal_vectors[(axis, cartesian)]
                    })
                    .sum();
                squared_length += component * component;
            }
            segment_lengths.push(squared_length.sqrt());
        }

        let mut node_distances = Vec::with_capacity(nodes.len());
        node_distances.push(0.0);
        for length in &segment_lengths {
            node_distances.push(node_distances.last().copied().unwrap_or_default() + length);
        }
        let total_length = *node_distances.last().unwrap_or(&0.0);
        if total_length == 0.0 {
            return Err(GeometryError::ZeroLengthPath);
        }

        let final_index = sample_count - 1;
        let node_indices: Vec<usize> = node_distances
            .iter()
            .map(|distance| ((distance / total_length) * final_index as f64).round() as usize)
            .collect();
        let mut reduced_points = vec![vec![0.0; periodic_dimension]; sample_count];
        let mut distances = vec![0.0; sample_count];

        for segment in 0..segment_lengths.len() {
            let start = node_indices[segment];
            let end = node_indices[segment + 1];
            let steps = end - start;
            for step in 0..=steps {
                let fraction = if steps == 0 {
                    0.0
                } else {
                    step as f64 / steps as f64
                };
                for component in 0..periodic_dimension {
                    reduced_points[start + step][component] = nodes[segment][component]
                        + fraction * (nodes[segment + 1][component] - nodes[segment][component]);
                }
                distances[start + step] =
                    node_distances[segment] + fraction * segment_lengths[segment];
            }
        }

        Ok(Self {
            reduced_points,
            distances,
            node_distances,
        })
    }

    /// Returns sampled momenta in reduced coordinates.
    #[must_use]
    pub fn reduced_points(&self) -> &[Vec<f64>] {
        &self.reduced_points
    }

    /// Returns cumulative Cartesian reciprocal-space distance per sample.
    #[must_use]
    pub fn distances(&self) -> &[f64] {
        &self.distances
    }

    /// Returns cumulative Cartesian reciprocal-space distance at each node.
    #[must_use]
    pub fn node_distances(&self) -> &[f64] {
        &self.node_distances
    }
}
