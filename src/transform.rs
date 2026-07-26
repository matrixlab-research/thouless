//! Structure-preserving transformations of tight-binding models.

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector};

use crate::model::{Lattice, ModelBuilder, OrbitalId, TightBindingModel};
use crate::ModelError;

const TRANSFORM_TOLERANCE: f64 = 1.0e-9;
const SUPERCELL_BOUNDARY_EPSILON: f64 = 1.414_213_562_373_095_2e-8;

/// A transformed supercell model and its old-cell representatives.
#[derive(Clone, Debug, PartialEq)]
pub struct SupercellModel {
    model: TightBindingModel,
    translations: Vec<Vec<i32>>,
}

impl SupercellModel {
    /// Returns the transformed tight-binding model.
    #[must_use]
    pub const fn model(&self) -> &TightBindingModel {
        &self.model
    }

    /// Returns old-lattice cell representatives contained in the supercell.
    #[must_use]
    pub fn translations(&self) -> &[Vec<i32>] {
        &self.translations
    }

    /// Consumes the result and returns the transformed model.
    #[must_use]
    pub fn into_model(self) -> TightBindingModel {
        self.model
    }
}

/// One source orbital in one integer lattice cell of a finite geometry.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct FiniteSite {
    cell: Vec<i32>,
    orbital: usize,
}

impl FiniteSite {
    /// Creates a site from a full-dimensional cell translation and source orbital.
    pub fn new(cell: impl IntoIterator<Item = i32>, orbital: usize) -> Self {
        Self {
            cell: cell.into_iter().collect(),
            orbital,
        }
    }

    /// Returns the source-cell translation.
    #[must_use]
    pub fn cell(&self) -> &[i32] {
        &self.cell
    }

    /// Returns the source orbital index.
    #[must_use]
    pub const fn orbital(&self) -> usize {
        self.orbital
    }
}

/// A finite model together with stable source-site provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteGeometry {
    model: TightBindingModel,
    sites: Vec<FiniteSite>,
}

impl FiniteGeometry {
    /// Returns the finite tight-binding model.
    #[must_use]
    pub const fn model(&self) -> &TightBindingModel {
        &self.model
    }

    /// Returns source sites in the same order as output orbitals.
    #[must_use]
    pub fn sites(&self) -> &[FiniteSite] {
        &self.sites
    }

    /// Consumes the result and returns the finite model.
    #[must_use]
    pub fn into_model(self) -> TightBindingModel {
        self.model
    }
}

/// Errors raised by tight-binding model transformations.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelTransformError {
    /// An orbital index is outside the source model.
    InvalidOrbital {
        /// Invalid orbital index.
        orbital: usize,
        /// Number of source orbitals.
        orbital_count: usize,
    },
    /// An orbital index occurs more than once.
    DuplicateOrbital {
        /// Repeated orbital index.
        orbital: usize,
    },
    /// A tight-binding model must retain at least one orbital.
    EmptyResult,
    /// A finite site has the wrong full cell dimension.
    InvalidFiniteCellDimension {
        /// Required number of integer coordinates.
        expected: usize,
        /// Number supplied.
        actual: usize,
    },
    /// The same finite source site occurs more than once.
    DuplicateFiniteSite,
    /// A real-space direction is outside the lattice.
    InvalidDirection {
        /// Invalid direction.
        direction: usize,
        /// Real-space dimension.
        dimension: usize,
    },
    /// Only a nonperiodic primitive vector may be changed.
    PeriodicDirection {
        /// Selected periodic direction.
        direction: usize,
    },
    /// A replacement primitive vector has the wrong dimension.
    InvalidVectorDimension {
        /// Required number of components.
        expected: usize,
        /// Supplied number of components.
        actual: usize,
    },
    /// A replacement primitive vector has zero numerical length.
    ZeroVector,
    /// A coordinate frame could not be inverted.
    SingularFrame,
    /// A hopping translation cannot be represented in the transformed frame.
    IncommensurateTranslation,
    /// A supercell matrix must have one square row per real-space direction.
    InvalidSupercellShape {
        /// Required square dimension.
        expected: usize,
    },
    /// A supercell may mix only directions that are both periodic.
    MixesOpenDirection,
    /// A supercell matrix is singular.
    SingularSupercell,
    /// A supercell matrix reverses orientation.
    LeftHandedSupercell,
    /// Integer cell representatives could not be enumerated consistently.
    SupercellEnumerationFailed {
        /// Number implied by the integer determinant.
        expected: usize,
        /// Number found by enumeration.
        actual: usize,
    },
    /// A transformed hopping did not map to an enumerated cell representative.
    MissingCellRepresentative,
    /// Rebuilding the transformed model violated a core model invariant.
    Model(ModelError),
}

impl std::fmt::Display for ModelTransformError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOrbital {
                orbital,
                orbital_count,
            } => write!(
                formatter,
                "orbital {orbital} is outside a model with {orbital_count} orbitals"
            ),
            Self::DuplicateOrbital { orbital } => {
                write!(formatter, "orbital {orbital} occurs more than once")
            }
            Self::EmptyResult => write!(formatter, "a model transformation removed every orbital"),
            Self::InvalidFiniteCellDimension { expected, actual } => write!(
                formatter,
                "finite site cell has {actual} coordinates; expected {expected}"
            ),
            Self::DuplicateFiniteSite => {
                write!(formatter, "finite source site occurs more than once")
            }
            Self::InvalidDirection {
                direction,
                dimension,
            } => write!(
                formatter,
                "real-space direction {direction} is outside dimension {dimension}"
            ),
            Self::PeriodicDirection { direction } => {
                write!(formatter, "direction {direction} is periodic")
            }
            Self::InvalidVectorDimension { expected, actual } => write!(
                formatter,
                "replacement vector has {actual} components; expected {expected}"
            ),
            Self::ZeroVector => write!(formatter, "replacement vector has zero length"),
            Self::SingularFrame => write!(formatter, "transformed coordinate frame is singular"),
            Self::IncommensurateTranslation => write!(
                formatter,
                "a hopping translation is not integral in the transformed frame"
            ),
            Self::InvalidSupercellShape { expected } => write!(
                formatter,
                "supercell matrix must have shape {expected}x{expected}"
            ),
            Self::MixesOpenDirection => write!(
                formatter,
                "supercell matrix may mix only directions that are both periodic"
            ),
            Self::SingularSupercell => write!(formatter, "supercell matrix is singular"),
            Self::LeftHandedSupercell => {
                write!(formatter, "supercell matrix must preserve orientation")
            }
            Self::SupercellEnumerationFailed { expected, actual } => write!(
                formatter,
                "supercell determinant requires {expected} representatives, found {actual}"
            ),
            Self::MissingCellRepresentative => {
                write!(formatter, "transformed hopping has no cell representative")
            }
            Self::Model(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ModelTransformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for ModelTransformError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

/// Extracts an arbitrary finite geometry from explicit source sites.
///
/// Every output orbital corresponds to one `(cell, orbital)` pair. Onsite
/// blocks are copied, a source hopping is retained only when both endpoints
/// occur in `sites`, and all output hopping translations are zero. This single
/// representation covers open boundaries, vacancies, holes, and shapes with
/// incomplete boundary cells.
pub fn make_finite_geometry(
    model: &TightBindingModel,
    sites: &[FiniteSite],
) -> Result<FiniteGeometry, ModelTransformError> {
    if sites.is_empty() {
        return Err(ModelTransformError::EmptyResult);
    }
    let dimension = model.lattice().real_dimension();
    let orbital_count = model.orbitals().len();
    let mut unique = HashSet::with_capacity(sites.len());
    for site in sites {
        if site.cell.len() != dimension {
            return Err(ModelTransformError::InvalidFiniteCellDimension {
                expected: dimension,
                actual: site.cell.len(),
            });
        }
        if site.orbital >= orbital_count {
            return Err(ModelTransformError::InvalidOrbital {
                orbital: site.orbital,
                orbital_count,
            });
        }
        if !unique.insert(site.clone()) {
            return Err(ModelTransformError::DuplicateFiniteSite);
        }
    }

    let lattice = Lattice::new(model.lattice().primitive_vectors().to_vec(), Vec::new())?;
    let mut builder = ModelBuilder::new(lattice);
    let mut output_orbitals = Vec::with_capacity(sites.len());
    let mut lookup = HashMap::with_capacity(sites.len());
    for (output_index, site) in sites.iter().enumerate() {
        let source = &model.orbitals()[site.orbital];
        let position = source
            .reduced_position()
            .iter()
            .zip(&site.cell)
            .map(|(position, cell)| position + f64::from(*cell))
            .collect::<Vec<_>>();
        let output = builder.add_orbital_with_dof(
            format!("{}@finite-{output_index}", source.label()),
            position,
            source.degrees_of_freedom(),
        )?;
        builder.set_onsite_block(output, model.onsite_blocks()[site.orbital].clone())?;
        output_orbitals.push(output);
        lookup.insert(site.clone(), output);
    }

    let zero_offset = vec![0; dimension];
    for (target_index, target_site) in sites.iter().enumerate() {
        for hopping in model
            .hoppings()
            .iter()
            .filter(|hopping| hopping.target().index() == target_site.orbital)
        {
            let source_site = FiniteSite {
                cell: target_site
                    .cell
                    .iter()
                    .zip(hopping.cell_offset())
                    .map(|(cell, offset)| cell + offset)
                    .collect(),
                orbital: hopping.source().index(),
            };
            let Some(source) = lookup.get(&source_site) else {
                continue;
            };
            builder.add_hopping_block_sum(
                output_orbitals[target_index],
                *source,
                zero_offset.iter().copied(),
                hopping.amplitude().clone(),
            )?;
        }
    }
    Ok(FiniteGeometry {
        model: builder.build()?,
        sites: sites.to_vec(),
    })
}

/// Extracts complete unit cells into a finite geometry.
pub fn make_finite_cluster(
    model: &TightBindingModel,
    cells: &[Vec<i32>],
) -> Result<FiniteGeometry, ModelTransformError> {
    let sites = cells
        .iter()
        .flat_map(|cell| {
            (0..model.orbitals().len())
                .map(move |orbital| FiniteSite::new(cell.iter().copied(), orbital))
        })
        .collect::<Vec<_>>();
    make_finite_geometry(model, &sites)
}

/// Replaces selected onsite blocks without changing geometry or hoppings.
///
/// This is the structural primitive for deterministic defects, random onsite
/// disorder, and spatially varying local fields. Replacements are indexed by
/// native orbital order and remain subject to the core shape and Hermiticity
/// invariants.
pub fn replace_onsite_blocks(
    model: &TightBindingModel,
    replacements: &[(usize, crate::ComplexMatrix)],
) -> Result<TightBindingModel, ModelTransformError> {
    let orbital_count = model.orbitals().len();
    let mut replacement_map = HashMap::with_capacity(replacements.len());
    for (orbital, block) in replacements {
        if *orbital >= orbital_count {
            return Err(ModelTransformError::InvalidOrbital {
                orbital: *orbital,
                orbital_count,
            });
        }
        if replacement_map.insert(*orbital, block).is_some() {
            return Err(ModelTransformError::DuplicateOrbital { orbital: *orbital });
        }
    }

    let mut builder = ModelBuilder::new(model.lattice().clone());
    let mut orbitals = Vec::with_capacity(orbital_count);
    for (index, source) in model.orbitals().iter().enumerate() {
        let output = builder.add_orbital_with_dof(
            source.label(),
            source.reduced_position().iter().copied(),
            source.degrees_of_freedom(),
        )?;
        builder.set_onsite_block(
            output,
            replacement_map.get(&index).map_or_else(
                || model.onsite_blocks()[index].clone(),
                |block| (*block).clone(),
            ),
        )?;
        orbitals.push(output);
    }
    for hopping in model.hoppings() {
        builder.add_hopping_block(
            orbitals[hopping.target().index()],
            orbitals[hopping.source().index()],
            hopping.cell_offset().iter().copied(),
            hopping.amplitude().clone(),
        )?;
    }
    builder.build().map_err(Into::into)
}

/// Returns a model with selected localized orbital subspaces removed.
///
/// Remaining orbitals preserve their relative order. Onsite blocks are copied,
/// hopping terms incident on removed orbitals are discarded, and all other
/// hopping endpoints are remapped to the compacted orbital index space.
pub fn remove_orbitals(
    model: &TightBindingModel,
    removed: &[usize],
) -> Result<TightBindingModel, ModelTransformError> {
    let orbital_count = model.orbitals().len();
    let mut removed_set = HashSet::with_capacity(removed.len());
    for &orbital in removed {
        if orbital >= orbital_count {
            return Err(ModelTransformError::InvalidOrbital {
                orbital,
                orbital_count,
            });
        }
        if !removed_set.insert(orbital) {
            return Err(ModelTransformError::DuplicateOrbital { orbital });
        }
    }
    if removed_set.len() == orbital_count {
        return Err(ModelTransformError::EmptyResult);
    }

    let mut builder = ModelBuilder::new(model.lattice().clone());
    let mut remapping: Vec<Option<OrbitalId>> = vec![None; orbital_count];
    for (index, orbital) in model.orbitals().iter().enumerate() {
        if removed_set.contains(&index) {
            continue;
        }
        let transformed = builder.add_orbital_with_dof(
            orbital.label(),
            orbital.reduced_position().iter().copied(),
            orbital.degrees_of_freedom(),
        )?;
        builder.set_onsite_block(transformed, model.onsite_blocks()[index].clone())?;
        remapping[index] = Some(transformed);
    }

    for hopping in model.hoppings() {
        let Some(target) = remapping[hopping.target().index()] else {
            continue;
        };
        let Some(source) = remapping[hopping.source().index()] else {
            continue;
        };
        builder.add_hopping_block(
            target,
            source,
            hopping.cell_offset().iter().copied(),
            hopping.amplitude().clone(),
        )?;
    }
    builder.build().map_err(Into::into)
}

/// Changes one nonperiodic primitive vector while preserving Cartesian geometry.
///
/// If `replacement` is absent, the old vector is projected out of the periodic
/// subspace and the remaining perpendicular component is rescaled to the old
/// vector length. Orbital coordinates and hopping translations are transformed
/// contravariantly. With `move_periodic_to_home`, periodic orbital coordinates
/// are shifted into `[0, 1)` and hopping offsets are adjusted exactly.
pub fn change_nonperiodic_vector(
    model: &TightBindingModel,
    direction: usize,
    replacement: Option<&[f64]>,
    move_periodic_to_home: bool,
) -> Result<TightBindingModel, ModelTransformError> {
    let dimension = model.lattice().real_dimension();
    if direction >= dimension {
        return Err(ModelTransformError::InvalidDirection {
            direction,
            dimension,
        });
    }
    if model.lattice().periodic_axes().contains(&direction) {
        return Err(ModelTransformError::PeriodicDirection { direction });
    }

    let old_primitive = matrix_from_rows(model.lattice().primitive_vectors());
    let old_vector = old_primitive.row(direction).transpose().into_owned();
    let new_vector = if let Some(vector) = replacement {
        if vector.len() != dimension {
            return Err(ModelTransformError::InvalidVectorDimension {
                expected: dimension,
                actual: vector.len(),
            });
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::NonFiniteValue {
                field: "replacement primitive vector",
            }
            .into());
        }
        DVector::from_column_slice(vector)
    } else {
        perpendicular_component(model.lattice(), &old_vector)?
    };
    let new_norm = new_vector.norm();
    if new_norm <= TRANSFORM_TOLERANCE {
        return Err(ModelTransformError::ZeroVector);
    }
    let new_vector = if replacement.is_none() {
        new_vector * (old_vector.norm() / new_norm)
    } else {
        new_vector
    };

    let mut new_primitive = old_primitive.clone();
    for component in 0..dimension {
        new_primitive[(direction, component)] = new_vector[component];
    }
    let new_inverse = new_primitive
        .clone()
        .try_inverse()
        .ok_or(ModelTransformError::SingularFrame)?;

    let positions = model
        .orbitals()
        .iter()
        .map(|orbital| {
            let reduced = DVector::from_column_slice(orbital.reduced_position());
            let cartesian = old_primitive.transpose() * reduced;
            let transformed = new_primitive
                .transpose()
                .try_inverse()
                .ok_or(ModelTransformError::SingularFrame)?
                * cartesian;
            Ok(transformed.iter().copied().collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>, ModelTransformError>>()?;

    let transformed_offsets = model
        .hoppings()
        .iter()
        .map(|hopping| {
            let old_offset = DVector::from_iterator(
                dimension,
                hopping.cell_offset().iter().map(|value| f64::from(*value)),
            );
            let cartesian = old_primitive.transpose() * old_offset;
            let transformed = new_inverse.transpose() * cartesian;
            transformed
                .iter()
                .map(|value| {
                    let rounded = value.round();
                    if (value - rounded).abs() > TRANSFORM_TOLERANCE {
                        return Err(ModelTransformError::IncommensurateTranslation);
                    }
                    Ok(rounded as i32)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let lattice = Lattice::new(
        rows_from_matrix(&new_primitive),
        model.lattice().periodic_axes().to_vec(),
    )?;
    rebuild_with_embedding(
        model,
        lattice,
        positions,
        transformed_offsets,
        move_periodic_to_home,
    )
}

/// Constructs a commensurate supercell of a periodic tight-binding model.
///
/// `integer_basis` stores the new primitive vectors as rows in the old reduced
/// basis. The returned translations are representatives of the quotient of old
/// lattice cells by the supercell lattice. Onsites, orbital embeddings, block
/// degrees of freedom, and hopping amplitudes are preserved exactly.
pub fn make_supercell(
    model: &TightBindingModel,
    integer_basis: &[Vec<i32>],
    move_periodic_to_home: bool,
) -> Result<SupercellModel, ModelTransformError> {
    let dimension = model.lattice().real_dimension();
    if dimension == 0
        || integer_basis.len() != dimension
        || integer_basis.iter().any(|row| row.len() != dimension)
    {
        return Err(ModelTransformError::InvalidSupercellShape {
            expected: dimension,
        });
    }
    let periodic = model.lattice().periodic_axes();
    for (row, basis_row) in integer_basis.iter().enumerate() {
        for (column, &entry) in basis_row.iter().enumerate() {
            if periodic.contains(&row) && periodic.contains(&column) {
                continue;
            }
            let required = if row == column { 1 } else { 0 };
            if entry != required {
                return Err(ModelTransformError::MixesOpenDirection);
            }
        }
    }

    let integer_matrix = DMatrix::from_row_slice(
        dimension,
        dimension,
        &integer_basis
            .iter()
            .flatten()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>(),
    );
    let determinant = integer_matrix.determinant();
    if determinant.abs() <= TRANSFORM_TOLERANCE {
        return Err(ModelTransformError::SingularSupercell);
    }
    if determinant < 0.0 {
        return Err(ModelTransformError::LeftHandedSupercell);
    }
    let expected_representatives = determinant.round() as usize;
    let inverse = integer_matrix
        .clone()
        .try_inverse()
        .ok_or(ModelTransformError::SingularSupercell)?;
    let translations = enumerate_supercell_translations(integer_basis, &inverse);
    if translations.len() != expected_representatives {
        return Err(ModelTransformError::SupercellEnumerationFailed {
            expected: expected_representatives,
            actual: translations.len(),
        });
    }

    let old_primitive = matrix_from_rows(model.lattice().primitive_vectors());
    let new_primitive = &integer_matrix * old_primitive;
    let lattice = Lattice::new(
        rows_from_matrix(&new_primitive),
        model.lattice().periodic_axes().to_vec(),
    )?;
    let source_orbital_count = model.orbitals().len();
    let expanded_count = translations.len().checked_mul(source_orbital_count).ok_or(
        ModelTransformError::SupercellEnumerationFailed {
            expected: expected_representatives,
            actual: 0,
        },
    )?;

    let mut positions = Vec::with_capacity(expanded_count);
    for translation in &translations {
        let translation =
            DVector::from_iterator(dimension, translation.iter().map(|value| f64::from(*value)));
        for orbital in model.orbitals() {
            let old_position = DVector::from_column_slice(orbital.reduced_position());
            let new_position = inverse.transpose() * (&translation + old_position);
            positions.push(new_position.iter().copied().collect::<Vec<_>>());
        }
    }
    let mut shifts = vec![vec![0_i32; dimension]; expanded_count];
    if move_periodic_to_home {
        for (position, shift) in positions.iter_mut().zip(&mut shifts) {
            for &axis in periodic {
                shift[axis] = position[axis].floor() as i32;
                position[axis] -= f64::from(shift[axis]);
            }
        }
    }

    let mut builder = ModelBuilder::new(lattice);
    let mut orbitals = Vec::with_capacity(expanded_count);
    for (translation_index, positions_for_cell) in
        positions.chunks(source_orbital_count).enumerate()
    {
        for (source_index, (source, position)) in
            model.orbitals().iter().zip(positions_for_cell).enumerate()
        {
            let transformed = builder.add_orbital_with_dof(
                format!("{}@{translation_index}", source.label()),
                position.iter().copied(),
                source.degrees_of_freedom(),
            )?;
            builder.set_onsite_block(transformed, model.onsite_blocks()[source_index].clone())?;
            orbitals.push(transformed);
        }
    }

    for (translation_index, translation) in translations.iter().enumerate() {
        for hopping in model.hoppings() {
            let total = DVector::from_iterator(
                dimension,
                translation
                    .iter()
                    .zip(hopping.cell_offset())
                    .map(|(cell, offset)| f64::from(*cell + *offset)),
            );
            let reduced = inverse.transpose() * &total;
            let supercell_offset = reduced
                .iter()
                .map(|value| (value + TRANSFORM_TOLERANCE).floor() as i32)
                .collect::<Vec<_>>();
            let supercell_offset_vector = DVector::from_iterator(
                dimension,
                supercell_offset.iter().map(|value| f64::from(*value)),
            );
            let representative = total - integer_matrix.transpose() * supercell_offset_vector;
            let representative = representative
                .iter()
                .map(|value| value.round() as i32)
                .collect::<Vec<_>>();
            let paired_translation = translations
                .iter()
                .position(|candidate| *candidate == representative)
                .ok_or(ModelTransformError::MissingCellRepresentative)?;

            let target_index = translation_index * source_orbital_count + hopping.target().index();
            let source_index = paired_translation * source_orbital_count + hopping.source().index();
            let mut offset = supercell_offset;
            if move_periodic_to_home {
                for ((component, &source_shift), &target_shift) in offset
                    .iter_mut()
                    .zip(&shifts[source_index])
                    .zip(&shifts[target_index])
                {
                    *component += source_shift - target_shift;
                }
            }
            builder.add_hopping_block_sum(
                orbitals[target_index],
                orbitals[source_index],
                offset,
                hopping.amplitude().clone(),
            )?;
        }
    }

    Ok(SupercellModel {
        model: builder.build()?,
        translations,
    })
}

fn enumerate_supercell_translations(
    integer_basis: &[Vec<i32>],
    inverse: &DMatrix<f64>,
) -> Vec<Vec<i32>> {
    let dimension = integer_basis.len();
    let bound = integer_basis
        .iter()
        .flatten()
        .map(|value| value.abs())
        .max()
        .unwrap_or(0) as usize
        * dimension;
    let mut translations = Vec::new();
    let mut candidate = vec![0_i32; dimension];
    enumerate_candidates(
        0,
        -(bound as i32),
        bound as i32,
        &mut candidate,
        inverse,
        &mut translations,
    );
    translations
}

fn enumerate_candidates(
    axis: usize,
    minimum: i32,
    maximum: i32,
    candidate: &mut [i32],
    inverse: &DMatrix<f64>,
    translations: &mut Vec<Vec<i32>>,
) {
    if axis == candidate.len() {
        let vector = DVector::from_iterator(
            candidate.len(),
            candidate.iter().map(|value| f64::from(*value)),
        );
        let reduced = inverse.transpose() * vector;
        if reduced.iter().all(|value| {
            -SUPERCELL_BOUNDARY_EPSILON < *value && *value <= 1.0 - SUPERCELL_BOUNDARY_EPSILON
        }) {
            translations.push(candidate.to_vec());
        }
        return;
    }
    for value in minimum..=maximum {
        candidate[axis] = value;
        enumerate_candidates(axis + 1, minimum, maximum, candidate, inverse, translations);
    }
}

fn perpendicular_component(
    lattice: &Lattice,
    vector: &DVector<f64>,
) -> Result<DVector<f64>, ModelTransformError> {
    if lattice.periodic_axes().is_empty() {
        return Ok(vector.clone());
    }
    let dimension = lattice.real_dimension();
    let periodic_count = lattice.periodic_dimension();
    let mut entries = Vec::with_capacity(periodic_count * dimension);
    for &axis in lattice.periodic_axes() {
        entries.extend_from_slice(&lattice.primitive_vectors()[axis]);
    }
    let periodic = DMatrix::from_row_slice(periodic_count, dimension, &entries);
    let gram_inverse = (&periodic * periodic.transpose())
        .try_inverse()
        .ok_or(ModelTransformError::SingularFrame)?;
    let coefficients = gram_inverse * &periodic * vector;
    Ok(vector - periodic.transpose() * coefficients)
}

fn rebuild_with_embedding(
    model: &TightBindingModel,
    lattice: Lattice,
    mut positions: Vec<Vec<f64>>,
    mut hopping_offsets: Vec<Vec<i32>>,
    move_periodic_to_home: bool,
) -> Result<TightBindingModel, ModelTransformError> {
    let dimension = lattice.real_dimension();
    let mut shifts = vec![vec![0_i32; dimension]; positions.len()];
    if move_periodic_to_home {
        for (position, shift) in positions.iter_mut().zip(&mut shifts) {
            for &axis in lattice.periodic_axes() {
                shift[axis] = position[axis].floor() as i32;
                position[axis] -= f64::from(shift[axis]);
            }
        }
        for (hopping, offset) in model.hoppings().iter().zip(&mut hopping_offsets) {
            let target_shift = &shifts[hopping.target().index()];
            let source_shift = &shifts[hopping.source().index()];
            for axis in 0..dimension {
                offset[axis] += source_shift[axis] - target_shift[axis];
            }
        }
    }

    let mut builder = ModelBuilder::new(lattice);
    let mut orbitals = Vec::with_capacity(model.orbitals().len());
    for (index, (orbital, position)) in model.orbitals().iter().zip(positions).enumerate() {
        let transformed = builder.add_orbital_with_dof(
            orbital.label(),
            position,
            orbital.degrees_of_freedom(),
        )?;
        builder.set_onsite_block(transformed, model.onsite_blocks()[index].clone())?;
        orbitals.push(transformed);
    }
    for (hopping, offset) in model.hoppings().iter().zip(hopping_offsets) {
        builder.add_hopping_block(
            orbitals[hopping.target().index()],
            orbitals[hopping.source().index()],
            offset,
            hopping.amplitude().clone(),
        )?;
    }
    builder.build().map_err(Into::into)
}

fn matrix_from_rows(rows: &[Vec<f64>]) -> DMatrix<f64> {
    DMatrix::from_row_slice(
        rows.len(),
        rows.first().map_or(0, Vec::len),
        &rows.iter().flatten().copied().collect::<Vec<_>>(),
    )
}

fn rows_from_matrix(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[(row, column)])
                .collect()
        })
        .collect()
}
