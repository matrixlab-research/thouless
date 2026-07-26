//! Coordinate-independent finite-difference stencils for ordered operators.
//!
//! Symbolic frontends assign opaque identifiers to coefficient factors. This
//! module applies ordered momentum operators, tracks every coefficient and
//! wave-function shift, and returns an exact rational grid geometry. It does
//! not depend on a particular symbolic algebra or source-language API.

use std::fmt;

use crate::Complex64;

/// One factor in an ordered differential-operator product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DifferentialFactor {
    /// An opaque coefficient whose symbolic value is owned by the frontend.
    Coefficient(usize),
    /// A nonnegative power of `-i d/dx_axis`.
    Momentum { axis: usize, power: usize },
}

/// An exact shift measured in units of a discretization spacing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RationalShift {
    numerator: i32,
    denominator: u32,
}

impl RationalShift {
    /// Signed numerator of the reduced-grid shift.
    #[must_use]
    pub const fn numerator(self) -> i32 {
        self.numerator
    }

    /// Positive denominator of the reduced-grid shift.
    #[must_use]
    pub const fn denominator(self) -> u32 {
        self.denominator
    }
}

/// One coefficient factor together with the point where it is evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShiftedCoefficient {
    id: usize,
    shifts: Vec<RationalShift>,
}

impl ShiftedCoefficient {
    /// Frontend-owned coefficient identifier.
    #[must_use]
    pub const fn id(&self) -> usize {
        self.id
    }

    /// Coordinate shifts in discretized-axis order.
    #[must_use]
    pub fn shifts(&self) -> &[RationalShift] {
        &self.shifts
    }
}

/// One contribution to a finite-difference hopping.
#[derive(Clone, Debug, PartialEq)]
pub struct DifferentialStencilTerm {
    wave_offset: Vec<i32>,
    weight: Complex64,
    inverse_spacing_powers: Vec<u32>,
    coefficients: Vec<ShiftedCoefficient>,
}

impl DifferentialStencilTerm {
    /// Integer hopping offset after reducing the finite-difference grid.
    #[must_use]
    pub fn wave_offset(&self) -> &[i32] {
        &self.wave_offset
    }

    /// Dimensionless complex central-difference weight.
    #[must_use]
    pub const fn weight(&self) -> Complex64 {
        self.weight
    }

    /// Power of the inverse grid spacing for every coordinate.
    #[must_use]
    pub fn inverse_spacing_powers(&self) -> &[u32] {
        &self.inverse_spacing_powers
    }

    /// Ordered symbolic coefficient factors and their evaluation shifts.
    #[must_use]
    pub fn coefficients(&self) -> &[ShiftedCoefficient] {
        &self.coefficients
    }
}

/// Invalid differential-factor or stencil geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContinuumError {
    /// No coordinate axes were supplied.
    EmptyDimension,
    /// A momentum factor addresses an axis outside the supplied dimension.
    InvalidAxis { axis: usize, dimension: usize },
    /// A magnetic field value was NaN or infinite.
    NonFiniteField,
    /// A requested power or shift exceeded the integer representation.
    Overflow,
}

impl fmt::Display for ContinuumError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimension => write!(formatter, "a stencil needs at least one coordinate"),
            Self::InvalidAxis { axis, dimension } => write!(
                formatter,
                "momentum axis {axis} is outside stencil dimension {dimension}"
            ),
            Self::NonFiniteField => write!(formatter, "magnetic field must be finite"),
            Self::Overflow => write!(formatter, "finite-difference stencil overflowed"),
        }
    }
}

impl std::error::Error for ContinuumError {}

#[derive(Clone)]
struct RawCoefficient {
    id: usize,
    shifts: Vec<i32>,
}

#[derive(Clone)]
struct RawTerm {
    wave_offset: Vec<i32>,
    weight: Complex64,
    inverse_spacing_powers: Vec<u32>,
    coefficients: Vec<RawCoefficient>,
}

fn integer_gcd(mut left: i32, mut right: i32) -> i32 {
    left = left.abs();
    right = right.abs();
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Apply ordered momentum operators and construct a centered-difference stencil.
///
/// Factors are supplied in mathematical left-to-right order. Coefficients are
/// opaque and remain ordered, while momentum factors act on everything to
/// their right. Each axis is shortened by the common divisor of its generated
/// wave offsets, reproducing the nearest-grid representation of repeated
/// centered derivatives.
pub fn finite_difference_stencil(
    dimension: usize,
    factors: &[DifferentialFactor],
) -> Result<Vec<DifferentialStencilTerm>, ContinuumError> {
    if dimension == 0 {
        return Err(ContinuumError::EmptyDimension);
    }
    let mut terms = vec![RawTerm {
        wave_offset: vec![0; dimension],
        weight: Complex64::new(1.0, 0.0),
        inverse_spacing_powers: vec![0; dimension],
        coefficients: Vec::new(),
    }];

    for factor in factors.iter().rev() {
        match *factor {
            DifferentialFactor::Coefficient(id) => {
                for term in &mut terms {
                    term.coefficients.insert(
                        0,
                        RawCoefficient {
                            id,
                            shifts: vec![0; dimension],
                        },
                    );
                }
            }
            DifferentialFactor::Momentum { axis, power } => {
                if axis >= dimension {
                    return Err(ContinuumError::InvalidAxis { axis, dimension });
                }
                for _ in 0..power {
                    let mut differentiated = Vec::with_capacity(2 * terms.len());
                    for term in terms {
                        for direction in [1_i32, -1_i32] {
                            let mut shifted = term.clone();
                            shifted.wave_offset[axis] = shifted.wave_offset[axis]
                                .checked_add(direction)
                                .ok_or(ContinuumError::Overflow)?;
                            for coefficient in &mut shifted.coefficients {
                                coefficient.shifts[axis] = coefficient.shifts[axis]
                                    .checked_add(direction)
                                    .ok_or(ContinuumError::Overflow)?;
                            }
                            shifted.inverse_spacing_powers[axis] = shifted.inverse_spacing_powers
                                [axis]
                                .checked_add(1)
                                .ok_or(ContinuumError::Overflow)?;
                            shifted.weight *= if direction > 0 {
                                Complex64::new(0.0, -0.5)
                            } else {
                                Complex64::new(0.0, 0.5)
                            };
                            differentiated.push(shifted);
                        }
                    }
                    terms = differentiated;
                }
            }
        }
    }

    let compression = (0..dimension)
        .map(|axis| {
            terms.iter().fold(0, |divisor, term| {
                integer_gcd(divisor, term.wave_offset[axis])
            })
        })
        .map(|divisor| divisor.max(1))
        .collect::<Vec<_>>();

    terms
        .into_iter()
        .map(|term| {
            let mut weight = term.weight;
            for (axis, divisor) in compression.iter().copied().enumerate() {
                weight *= (divisor as f64).powi(
                    i32::try_from(term.inverse_spacing_powers[axis])
                        .map_err(|_| ContinuumError::Overflow)?,
                );
            }
            Ok(DifferentialStencilTerm {
                wave_offset: term
                    .wave_offset
                    .iter()
                    .zip(&compression)
                    .map(|(offset, divisor)| offset / divisor)
                    .collect(),
                weight,
                inverse_spacing_powers: term.inverse_spacing_powers,
                coefficients: term
                    .coefficients
                    .into_iter()
                    .map(|coefficient| ShiftedCoefficient {
                        id: coefficient.id,
                        shifts: coefficient
                            .shifts
                            .into_iter()
                            .zip(&compression)
                            .map(|(numerator, denominator)| RationalShift {
                                numerator,
                                denominator: *denominator as u32,
                            })
                            .collect(),
                    })
                    .collect(),
            })
        })
        .collect()
}

/// Evaluate an ordered product of Landau raising and lowering operators.
///
/// Positive entries encode powers of the raising operator and negative
/// entries powers of the lowering operator. Entries are ordered
/// left-to-right as written in the operator product. A negative magnetic
/// field exchanges raising and lowering operators.
pub fn landau_ladder_coefficient(
    ladder_powers: &[i32],
    initial_level: usize,
    magnetic_field: f64,
) -> Result<f64, ContinuumError> {
    if !magnetic_field.is_finite() {
        return Err(ContinuumError::NonFiniteField);
    }
    let mut level = initial_level;
    let mut coefficient = 1.0;
    for &encoded_power in ladder_powers.iter().rev() {
        let power = if magnetic_field < 0.0 {
            encoded_power
                .checked_neg()
                .ok_or(ContinuumError::Overflow)?
        } else {
            encoded_power
        };
        match power.cmp(&0) {
            std::cmp::Ordering::Greater => {
                for _ in 0..power {
                    level = level.checked_add(1).ok_or(ContinuumError::Overflow)?;
                    coefficient *= (level as f64).sqrt();
                }
            }
            std::cmp::Ordering::Less => {
                for _ in power..0 {
                    if level == 0 {
                        return Ok(0.0);
                    }
                    coefficient *= (level as f64).sqrt();
                    level -= 1;
                }
            }
            std::cmp::Ordering::Equal => {}
        }
    }
    Ok(coefficient)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rational(term: &DifferentialStencilTerm, coefficient: usize) -> RationalShift {
        term.coefficients()[coefficient].shifts()[0]
    }

    #[test]
    fn second_derivative_shortens_to_nearest_neighbors() {
        let terms =
            finite_difference_stencil(1, &[DifferentialFactor::Momentum { axis: 0, power: 2 }])
                .unwrap();
        let mut by_offset = std::collections::BTreeMap::new();
        for term in terms {
            *by_offset
                .entry(term.wave_offset()[0])
                .or_insert(Complex64::new(0.0, 0.0)) += term.weight();
            assert_eq!(term.inverse_spacing_powers(), &[2]);
        }
        assert_eq!(by_offset[&-1], Complex64::new(-1.0, 0.0));
        assert_eq!(by_offset[&0], Complex64::new(2.0, 0.0));
        assert_eq!(by_offset[&1], Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn divergence_form_places_coefficients_on_half_grid_links() {
        let terms = finite_difference_stencil(
            1,
            &[
                DifferentialFactor::Momentum { axis: 0, power: 1 },
                DifferentialFactor::Coefficient(7),
                DifferentialFactor::Momentum { axis: 0, power: 1 },
            ],
        )
        .unwrap();
        assert!(terms.iter().all(|term| term.coefficients()[0].id() == 7));
        let forward = terms.iter().find(|term| term.wave_offset() == [1]).unwrap();
        assert_eq!(forward.weight(), Complex64::new(-1.0, 0.0));
        assert_eq!(
            rational(forward, 0),
            RationalShift {
                numerator: 1,
                denominator: 2,
            }
        );
        let onsite_shifts = terms
            .iter()
            .filter(|term| term.wave_offset() == [0])
            .map(|term| rational(term, 0).numerator())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(onsite_shifts, [-1, 1].into_iter().collect());
    }

    #[test]
    fn ladder_products_respect_order_truncation_and_field_orientation() {
        let product = landau_ladder_coefficient(&[-2, 3, -2], 5, 1.0).unwrap();
        assert!((product - 4.0 * 5.0 * 6.0 * 5.0_f64.sqrt()).abs() < 1.0e-12);
        assert_eq!(
            landau_ladder_coefficient(&[-3, -2, 3], 1, 1.0).unwrap(),
            0.0
        );
        assert_eq!(
            landau_ladder_coefficient(&[1, -1], 2, 1.0).unwrap(),
            landau_ladder_coefficient(&[-1, 1], 2, -1.0).unwrap()
        );
    }
}
