//! Compact-support interpolation of discrete densities and bond currents.

use std::fmt;

const CROSS_SECTIONS: [f64; 3] = [
    16.0 / 15.0,
    std::f64::consts::PI / 3.0,
    0.957_437_761_094_032,
];

/// Errors raised while constructing a regularized field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterpolationError {
    /// Coordinates, values, or dimensions do not agree.
    InvalidShape,
    /// A width, resolution, coordinate, or field value is invalid.
    InvalidValue,
}

impl fmt::Display for InterpolationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                write!(formatter, "interpolation inputs have incompatible shapes")
            }
            Self::InvalidValue => {
                write!(formatter, "interpolation inputs contain an invalid value")
            }
        }
    }
}

impl std::error::Error for InterpolationError {}

/// Resolution and smoothing-width policy for a regularized field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothingOptions {
    /// Absolute compact-support diameter. Takes precedence when supplied.
    pub absolute_width: Option<f64>,
    /// Diameter relative to the longest side of the point bounding box.
    pub relative_width: Option<f64>,
    /// Number of grid intervals sampled across one smoothing diameter.
    pub samples_per_width: usize,
}

impl Default for SmoothingOptions {
    fn default() -> Self {
        Self {
            absolute_width: None,
            relative_width: None,
            samples_per_width: 9,
        }
    }
}

/// A scalar or vector field sampled on a Cartesian grid in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct RegularField {
    shape: Vec<usize>,
    components: usize,
    bounds: Vec<(f64, f64)>,
    values: Vec<f64>,
}

impl RegularField {
    /// Spatial grid shape, excluding the component axis.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of scalar components stored at every grid point.
    #[must_use]
    pub fn components(&self) -> usize {
        self.components
    }

    /// Inclusive coordinate bounds for every Cartesian axis.
    #[must_use]
    pub fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    /// Row-major samples with the component axis varying fastest.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

fn validate_points(points: &[Vec<f64>]) -> Result<usize, InterpolationError> {
    let dimension = points.first().map_or(0, Vec::len);
    if points.is_empty()
        || !matches!(dimension, 1..=3)
        || points.iter().any(|point| point.len() != dimension)
    {
        return Err(InterpolationError::InvalidShape);
    }
    if points
        .iter()
        .flatten()
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(InterpolationError::InvalidValue);
    }
    Ok(dimension)
}

fn bounding_box(points: &[Vec<f64>], dimension: usize) -> (Vec<f64>, Vec<f64>) {
    let mut minimum = points[0].clone();
    let mut maximum = points[0].clone();
    for point in &points[1..] {
        for axis in 0..dimension {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    (minimum, maximum)
}

fn resolve_width(
    lengths: &[f64],
    box_size: &[f64],
    options: SmoothingOptions,
) -> Result<f64, InterpolationError> {
    if options.samples_per_width < 2
        || options
            .absolute_width
            .is_some_and(|width| !width.is_finite() || width <= 0.0)
        || options
            .relative_width
            .is_some_and(|width| !width.is_finite() || width <= 0.0)
    {
        return Err(InterpolationError::InvalidValue);
    }
    let width = if let Some(width) = options.absolute_width {
        width
    } else if let Some(relative_width) = options.relative_width {
        relative_width * box_size.iter().copied().fold(0.0, f64::max)
    } else {
        let mut positive = lengths
            .iter()
            .copied()
            .filter(|length| length.is_finite() && *length > 0.0)
            .collect::<Vec<_>>();
        positive.sort_by(f64::total_cmp);
        let longest = positive
            .last()
            .copied()
            .ok_or(InterpolationError::InvalidValue)?;
        let shortest = positive
            .into_iter()
            .find(|length| *length / longest > 1.0e-3)
            .ok_or(InterpolationError::InvalidValue)?;
        4.0 * shortest
    };
    if !width.is_finite() || width <= 0.0 {
        return Err(InterpolationError::InvalidValue);
    }
    Ok(width)
}

fn grid_geometry(
    minimum: &[f64],
    maximum: &[f64],
    width: f64,
    samples_per_width: usize,
) -> (Vec<usize>, Vec<(f64, f64)>) {
    let padding = width / 2.0;
    let shape = minimum
        .iter()
        .zip(maximum)
        .map(|(minimum, maximum)| {
            let mut size = (((maximum - minimum) * samples_per_width as f64 / width)
                + samples_per_width as f64) as usize;
            size = size.max(2);
            if size % 2 == 1 {
                size += 1;
            }
            size
        })
        .collect::<Vec<_>>();
    let bounds = minimum
        .iter()
        .zip(maximum)
        .map(|(minimum, maximum)| (minimum - padding, maximum + padding))
        .collect();
    (shape, bounds)
}

fn grid_axes(shape: &[usize], bounds: &[(f64, f64)]) -> Vec<Vec<f64>> {
    shape
        .iter()
        .enumerate()
        .map(|(axis, size)| {
            let (minimum, maximum) = bounds[axis];
            (0..*size)
                .map(|index| minimum + index as f64 * (maximum - minimum) / (*size - 1) as f64)
                .collect()
        })
        .collect()
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn support_ranges(
    element_minimum: &[f64],
    element_maximum: &[f64],
    box_minimum: &[f64],
    box_maximum: &[f64],
    width: f64,
    shape: &[usize],
) -> Vec<std::ops::Range<usize>> {
    (0..shape.len())
        .map(|axis| {
            let density =
                (shape[axis] - 1) as f64 / (box_maximum[axis] + width - box_minimum[axis]);
            let start = ((element_minimum[axis] - box_minimum[axis]) * density)
                .floor()
                .max(0.0) as usize;
            let end = ((element_maximum[axis] + width - box_minimum[axis]) * density)
                .ceil()
                .max(0.0) as usize;
            start.min(shape[axis])..end.min(shape[axis])
        })
        .collect()
}

fn visit_range_indices(
    ranges: &[std::ops::Range<usize>],
    strides: &[usize],
    axis: usize,
    linear_index: usize,
    indices: &mut [usize],
    visitor: &mut impl FnMut(usize, &[usize]),
) {
    if axis == ranges.len() {
        visitor(linear_index, indices);
        return;
    }
    for index in ranges[axis].clone() {
        indices[axis] = index;
        visit_range_indices(
            ranges,
            strides,
            axis + 1,
            linear_index + index * strides[axis],
            indices,
            visitor,
        );
    }
}

fn bump(radius_squared: f64) -> f64 {
    if radius_squared >= 1.0 {
        0.0
    } else {
        (1.0 - radius_squared).powi(2)
    }
}

fn smoothing(radial_distance: f64, axial_distance: f64) -> f64 {
    let radial_radius_squared = (1.0 - radial_distance * radial_distance).max(0.0);
    let radial_radius = radial_radius_squared.sqrt();
    let clipped_axial = axial_distance.clamp(-radial_radius, radial_radius);
    let axial_squared = clipped_axial * clipped_axial;
    let radial_fourth = radial_radius_squared * radial_radius_squared;
    clipped_axial
        * (axial_squared * (axial_squared / 5.0 - (2.0 / 3.0) * radial_radius_squared)
            + radial_fourth)
        + (8.0 / 15.0) * radial_fourth * radial_radius
}

fn edge_lengths(edges: &[(Vec<f64>, Vec<f64>)]) -> Vec<f64> {
    edges
        .iter()
        .map(|(first, second)| {
            first
                .iter()
                .zip(second)
                .map(|(first, second)| (second - first).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

/// Smooth point-supported scalar values with a compact quartic bump.
pub fn interpolate_density(
    points: &[Vec<f64>],
    values: &[f64],
    reference_edges: &[(Vec<f64>, Vec<f64>)],
    options: SmoothingOptions,
) -> Result<RegularField, InterpolationError> {
    let dimension = validate_points(points)?;
    if values.len() != points.len()
        || values.iter().any(|value| !value.is_finite())
        || reference_edges.iter().any(|(first, second)| {
            first.len() != dimension
                || second.len() != dimension
                || first
                    .iter()
                    .chain(second)
                    .any(|coordinate| !coordinate.is_finite())
        })
    {
        return Err(InterpolationError::InvalidShape);
    }
    let (minimum, maximum) = bounding_box(points, dimension);
    let box_size = maximum
        .iter()
        .zip(&minimum)
        .map(|(maximum, minimum)| maximum - minimum)
        .collect::<Vec<_>>();
    let width = resolve_width(&edge_lengths(reference_edges), &box_size, options)?;
    let (shape, bounds) = grid_geometry(&minimum, &maximum, width, options.samples_per_width);
    let sample_count = shape.iter().product::<usize>();
    let scale = 2.0 / width;
    let mut field = vec![0.0; sample_count];
    let axes = grid_axes(&shape, &bounds);
    let strides = row_major_strides(&shape);
    for (point, value) in points.iter().zip(values) {
        let ranges = support_ranges(point, point, &minimum, &maximum, width, &shape);
        let mut indices = vec![0; dimension];
        visit_range_indices(
            &ranges,
            &strides,
            0,
            0,
            &mut indices,
            &mut |linear_index, indices| {
                let radius_squared = (0..dimension)
                    .map(|axis| ((axes[axis][indices[axis]] - point[axis]) * scale).powi(2))
                    .sum();
                field[linear_index] += value * bump(radius_squared);
            },
        );
    }
    let normalization = scale / CROSS_SECTIONS[dimension - 1];
    for value in &mut field {
        *value *= normalization;
    }
    Ok(RegularField {
        shape,
        components: 1,
        bounds,
        values: field,
    })
}

/// Smooth oriented bond currents into a continuous vector field.
pub fn interpolate_current(
    edges: &[(Vec<f64>, Vec<f64>)],
    currents: &[f64],
    options: SmoothingOptions,
) -> Result<RegularField, InterpolationError> {
    let points = edges
        .iter()
        .flat_map(|(first, second)| [first.clone(), second.clone()])
        .collect::<Vec<_>>();
    let dimension = validate_points(&points)?;
    if currents.len() != edges.len()
        || currents.iter().any(|value| !value.is_finite())
        || edges
            .iter()
            .any(|(first, second)| first.len() != dimension || second.len() != dimension)
    {
        return Err(InterpolationError::InvalidShape);
    }
    let lengths = edge_lengths(edges);
    if lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= 0.0)
    {
        return Err(InterpolationError::InvalidValue);
    }
    let (minimum, maximum) = bounding_box(&points, dimension);
    let box_size = maximum
        .iter()
        .zip(&minimum)
        .map(|(maximum, minimum)| maximum - minimum)
        .collect::<Vec<_>>();
    let width = resolve_width(&lengths, &box_size, options)?;
    let (shape, bounds) = grid_geometry(&minimum, &maximum, width, options.samples_per_width);
    let sample_count = shape.iter().product::<usize>();
    let scale = 2.0 / width;
    let directions = edges
        .iter()
        .zip(&lengths)
        .map(|((first, second), length)| {
            first
                .iter()
                .zip(second)
                .map(|(first, second)| (second - first) / length)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut field = vec![0.0; sample_count * dimension];
    let axes = grid_axes(&shape, &bounds);
    let strides = row_major_strides(&shape);
    for (((first, second), length), (direction, current)) in edges
        .iter()
        .zip(&lengths)
        .zip(directions.iter().zip(currents))
    {
        let element_minimum = first
            .iter()
            .zip(second)
            .map(|(first, second)| first.min(*second))
            .collect::<Vec<_>>();
        let element_maximum = first
            .iter()
            .zip(second)
            .map(|(first, second)| first.max(*second))
            .collect::<Vec<_>>();
        let ranges = support_ranges(
            &element_minimum,
            &element_maximum,
            &minimum,
            &maximum,
            width,
            &shape,
        );
        let mut indices = vec![0; dimension];
        visit_range_indices(
            &ranges,
            &strides,
            0,
            0,
            &mut indices,
            &mut |linear_index, indices| {
                let displacement = (0..dimension)
                    .map(|axis| (axes[axis][indices[axis]] - first[axis]) * scale)
                    .collect::<Vec<_>>();
                let axial = displacement
                    .iter()
                    .zip(direction)
                    .map(|(coordinate, direction)| coordinate * direction)
                    .sum::<f64>();
                let radius_squared = displacement.iter().map(|value| value * value).sum::<f64>();
                let radial = (radius_squared - axial * axial).abs().sqrt();
                let magnitude = current
                    * (smoothing(radial, axial) - smoothing(radial, axial - length * scale));
                for component in 0..dimension {
                    field[linear_index * dimension + component] += direction[component] * magnitude;
                }
            },
        );
    }
    let normalization = scale / CROSS_SECTIONS[dimension - 1];
    for value in &mut field {
        *value *= normalization;
    }
    Ok(RegularField {
        shape,
        components: dimension,
        bounds,
        values: field,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(width: f64, samples_per_width: usize) -> SmoothingOptions {
        SmoothingOptions {
            absolute_width: Some(width),
            relative_width: None,
            samples_per_width,
        }
    }

    #[test]
    fn density_interpolation_is_linear_and_zero_on_the_padded_border() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let edges = vec![(points[0].clone(), points[1].clone())];
        let first = interpolate_density(&points, &[1.0, 0.5], &edges, options(1.0, 10))
            .expect("valid field");
        let second = interpolate_density(&points, &[-0.5, 2.0], &edges, options(1.0, 10))
            .expect("valid field");
        let combined = interpolate_density(&points, &[0.0, 4.5], &edges, options(1.0, 10))
            .expect("valid field");
        for ((first, second), combined) in first
            .values()
            .iter()
            .zip(second.values())
            .zip(combined.values())
        {
            assert!((first + 2.0 * second - combined).abs() < 1.0e-12);
        }
        let rows = first.shape()[0];
        let columns = first.shape()[1];
        for row in 0..rows {
            for column in 0..columns {
                if row == 0 || row + 1 == rows || column == 0 || column + 1 == columns {
                    assert_eq!(first.values()[row * columns + column], 0.0);
                }
            }
        }
    }

    #[test]
    fn bond_current_points_along_the_oriented_edge_and_has_zero_border() {
        let edge = (vec![0.0, 0.0], vec![1.0, 0.0]);
        let field = interpolate_current(&[edge], &[2.0], options(1.0, 10)).expect("valid field");
        assert_eq!(field.components(), 2);
        assert!(field
            .values()
            .chunks_exact(2)
            .all(|sample| sample[1].abs() < 1.0e-12));
        assert!(field.values().chunks_exact(2).any(|sample| sample[0] > 0.0));
        let rows = field.shape()[0];
        let columns = field.shape()[1];
        for row in 0..rows {
            for column in 0..columns {
                if row == 0 || row + 1 == rows || column == 0 || column + 1 == columns {
                    let offset = (row * columns + column) * 2;
                    assert!(field.values()[offset].abs() < 1.0e-12);
                    assert!(field.values()[offset + 1].abs() < 1.0e-12);
                }
            }
        }
    }
}
