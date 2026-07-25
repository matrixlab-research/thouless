//! Geometry transformations and sampling in real and reciprocal space.

use std::f64::consts::TAU;

use nalgebra::DMatrix;

use crate::model::Lattice;
use crate::GeometryError;

/// Uniform tensor-product sampling of one primitive reciprocal cell.
///
/// Points are ordered lexicographically with the last reduced-coordinate axis
/// varying fastest. A fractional offset of `0` includes the reciprocal origin;
/// `0.5` places points at cell centers along that axis.
#[derive(Clone, Debug, PartialEq)]
pub struct UniformReciprocalMesh {
    shape: Vec<usize>,
    fractional_offsets: Vec<f64>,
    reduced_points: Vec<Vec<f64>>,
    cartesian_points: Vec<Vec<f64>>,
    reciprocal_volume: f64,
}

impl UniformReciprocalMesh {
    /// Creates a uniform reciprocal mesh.
    ///
    /// `shape` and `fractional_offsets` contain one entry per periodic lattice
    /// direction. Coordinates are `(index + offset) / extent`, so every point
    /// belongs to the half-open reduced primitive cell `[0, 1)^d`.
    pub fn new(
        lattice: &Lattice,
        shape: &[usize],
        fractional_offsets: &[f64],
    ) -> Result<Self, GeometryError> {
        let periodic_dimension = lattice.periodic_dimension();
        if periodic_dimension == 0 {
            return Err(GeometryError::NoPeriodicDirections);
        }
        if shape.len() != periodic_dimension {
            return Err(GeometryError::InvalidMeshShape {
                expected: periodic_dimension,
                actual: shape.len(),
            });
        }
        if fractional_offsets.len() != periodic_dimension {
            return Err(GeometryError::InvalidMeshShape {
                expected: periodic_dimension,
                actual: fractional_offsets.len(),
            });
        }
        if let Some(axis) = shape.iter().position(|extent| *extent == 0) {
            return Err(GeometryError::EmptyMeshAxis { axis });
        }
        if let Some(axis) = fractional_offsets
            .iter()
            .position(|offset| !offset.is_finite() || !(0.0..1.0).contains(offset))
        {
            return Err(GeometryError::InvalidMeshOffset { axis });
        }
        let point_count = shape
            .iter()
            .try_fold(1_usize, |count, extent| count.checked_mul(*extent));
        let point_count = point_count.ok_or(GeometryError::MeshSizeOverflow)?;
        let reciprocal_vectors = reciprocal_vectors(lattice)?;
        let reciprocal_gram = &reciprocal_vectors * reciprocal_vectors.transpose();
        let reciprocal_volume = reciprocal_gram.determinant().max(0.0).sqrt();
        if !reciprocal_volume.is_finite() || reciprocal_volume == 0.0 {
            return Err(GeometryError::SingularPeriodicGeometry);
        }

        let mut reduced_points = Vec::with_capacity(point_count);
        let mut cartesian_points = Vec::with_capacity(point_count);
        for flat_index in 0..point_count {
            let mut remainder = flat_index;
            let mut indices = vec![0; periodic_dimension];
            for axis in (0..periodic_dimension).rev() {
                indices[axis] = remainder % shape[axis];
                remainder /= shape[axis];
            }
            let reduced = indices
                .iter()
                .zip(shape)
                .zip(fractional_offsets)
                .map(|((&index, &extent), &offset)| (index as f64 + offset) / extent as f64)
                .collect::<Vec<_>>();
            let cartesian = (0..lattice.real_dimension())
                .map(|component| {
                    reduced
                        .iter()
                        .enumerate()
                        .map(|(axis, coordinate)| {
                            coordinate * reciprocal_vectors[(axis, component)]
                        })
                        .sum()
                })
                .collect();
            reduced_points.push(reduced);
            cartesian_points.push(cartesian);
        }

        Ok(Self {
            shape: shape.to_vec(),
            fractional_offsets: fractional_offsets.to_vec(),
            reduced_points,
            cartesian_points,
            reciprocal_volume,
        })
    }

    /// Creates a mesh that includes the reciprocal origin on every axis.
    pub fn gamma_centered(lattice: &Lattice, shape: &[usize]) -> Result<Self, GeometryError> {
        Self::new(lattice, shape, &vec![0.0; shape.len()])
    }

    /// Creates a mesh shifted to the center of every reciprocal grid cell.
    pub fn cell_centered(lattice: &Lattice, shape: &[usize]) -> Result<Self, GeometryError> {
        Self::new(lattice, shape, &vec![0.5; shape.len()])
    }

    /// Returns the tensor-product mesh extents.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns fractional offsets measured in grid cells.
    #[must_use]
    pub fn fractional_offsets(&self) -> &[f64] {
        &self.fractional_offsets
    }

    /// Returns all momenta in reduced coordinates.
    #[must_use]
    pub fn reduced_points(&self) -> &[Vec<f64>] {
        &self.reduced_points
    }

    /// Returns all momenta in Cartesian reciprocal coordinates.
    #[must_use]
    pub fn cartesian_points(&self) -> &[Vec<f64>] {
        &self.cartesian_points
    }

    /// Returns the measure of the primitive reciprocal cell.
    #[must_use]
    pub const fn reciprocal_volume(&self) -> f64 {
        self.reciprocal_volume
    }

    /// Returns the constant quadrature weight normalized to sum to one.
    #[must_use]
    pub fn normalized_weight(&self) -> f64 {
        1.0 / self.reduced_points.len() as f64
    }

    /// Returns the constant quadrature weight in Cartesian reciprocal measure.
    #[must_use]
    pub fn cartesian_weight(&self) -> f64 {
        self.reciprocal_volume * self.normalized_weight()
    }
}

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
        let reciprocal_vectors = reciprocal_vectors(lattice)?;

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
        let mut node_indices = node_distances
            .iter()
            .map(|distance| ((distance / total_length) * final_index as f64).round() as usize)
            .collect::<Vec<_>>();
        for node in 1..(nodes.len() - 1) {
            let minimum = node_indices[node - 1] + 1;
            let maximum = final_index - (nodes.len() - 1 - node);
            node_indices[node] = node_indices[node].clamp(minimum, maximum);
        }
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

fn reciprocal_vectors(lattice: &Lattice) -> Result<DMatrix<f64>, GeometryError> {
    let periodic_dimension = lattice.periodic_dimension();
    let real_dimension = lattice.real_dimension();
    let periodic_vectors =
        DMatrix::from_fn(periodic_dimension, real_dimension, |periodic, cartesian| {
            lattice.primitive_vectors()[lattice.periodic_axes()[periodic]][cartesian]
        });
    let gram = &periodic_vectors * periodic_vectors.transpose();
    Ok(gram
        .try_inverse()
        .ok_or(GeometryError::SingularPeriodicGeometry)?
        * periodic_vectors
        * TAU)
}
