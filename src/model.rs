//! Domain objects and invariants shared by periodic, finite, and open systems.

use crate::{Complex64, ModelError};

/// Translation structure embedded in real space.
///
/// A finite system has zero translation vectors. A wire or slab may have fewer
/// translation vectors than embedding dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct Lattice {
    real_dimension: usize,
    translation_vectors: Vec<Vec<f64>>,
}

impl Lattice {
    /// Creates a lattice from an embedding dimension and translation vectors.
    pub fn new(
        real_dimension: usize,
        translation_vectors: Vec<Vec<f64>>,
    ) -> Result<Self, ModelError> {
        if real_dimension == 0 {
            return Err(ModelError::InvalidRealDimension);
        }
        if translation_vectors.len() > real_dimension {
            return Err(ModelError::TooManyTranslationVectors {
                real_dimension,
                translation_count: translation_vectors.len(),
            });
        }
        for (index, vector) in translation_vectors.iter().enumerate() {
            if vector.len() != real_dimension {
                return Err(ModelError::InvalidTranslationVector {
                    index,
                    expected: real_dimension,
                    actual: vector.len(),
                });
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(ModelError::NonFiniteValue {
                    field: "translation vector",
                });
            }
        }
        Ok(Self {
            real_dimension,
            translation_vectors,
        })
    }

    /// Returns the dimension of the embedding space.
    #[must_use]
    pub const fn real_dimension(&self) -> usize {
        self.real_dimension
    }

    /// Returns the number of periodic translation directions.
    #[must_use]
    pub fn periodic_dimension(&self) -> usize {
        self.translation_vectors.len()
    }

    /// Returns the primitive translation vectors.
    #[must_use]
    pub fn translation_vectors(&self) -> &[Vec<f64>] {
        &self.translation_vectors
    }
}

/// Stable identifier for an orbital within one model.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct OrbitalId(usize);

impl OrbitalId {
    /// Returns the zero-based orbital index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// An orbital and its position inside the reference cell.
#[derive(Clone, Debug, PartialEq)]
pub struct Orbital {
    label: String,
    position: Vec<f64>,
}

impl Orbital {
    /// Returns the user-visible orbital label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the embedding-space position.
    #[must_use]
    pub fn position(&self) -> &[f64] {
        &self.position
    }
}

/// A directed hopping whose Hermitian conjugate is implicit.
#[derive(Clone, Debug, PartialEq)]
pub struct Hopping {
    from: OrbitalId,
    to: OrbitalId,
    cell_offset: Vec<i32>,
    amplitude: Complex64,
}

impl Hopping {
    /// Returns the source orbital.
    #[must_use]
    pub const fn from(&self) -> OrbitalId {
        self.from
    }

    /// Returns the destination orbital.
    #[must_use]
    pub const fn to(&self) -> OrbitalId {
        self.to
    }

    /// Returns the integer displacement in primitive-cell coordinates.
    #[must_use]
    pub fn cell_offset(&self) -> &[i32] {
        &self.cell_offset
    }

    /// Returns the complex hopping amplitude.
    #[must_use]
    pub const fn amplitude(&self) -> Complex64 {
        self.amplitude
    }
}

/// Builder for the structural part of a tight-binding Hamiltonian.
#[derive(Clone, Debug)]
pub struct ModelBuilder {
    lattice: Lattice,
    orbitals: Vec<Orbital>,
    onsite: Vec<f64>,
    hoppings: Vec<Hopping>,
}

impl ModelBuilder {
    /// Starts a model in the supplied translation structure.
    #[must_use]
    pub fn new(lattice: Lattice) -> Self {
        Self {
            lattice,
            orbitals: Vec::new(),
            onsite: Vec::new(),
            hoppings: Vec::new(),
        }
    }

    /// Adds an orbital and returns its stable model-local identifier.
    pub fn add_orbital(
        &mut self,
        label: impl Into<String>,
        position: impl IntoIterator<Item = f64>,
    ) -> Result<OrbitalId, ModelError> {
        let label = label.into();
        if label.is_empty() {
            return Err(ModelError::EmptyOrbitalLabel);
        }
        if self.orbitals.iter().any(|orbital| orbital.label == label) {
            return Err(ModelError::DuplicateOrbitalLabel { label });
        }

        let position: Vec<f64> = position.into_iter().collect();
        if position.len() != self.lattice.real_dimension {
            return Err(ModelError::InvalidOrbitalPosition {
                expected: self.lattice.real_dimension,
                actual: position.len(),
            });
        }
        if position.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::NonFiniteValue {
                field: "orbital position",
            });
        }

        let id = OrbitalId(self.orbitals.len());
        self.orbitals.push(Orbital { label, position });
        self.onsite.push(0.0);
        Ok(id)
    }

    /// Sets a real scalar onsite energy.
    pub fn set_onsite(&mut self, orbital: OrbitalId, energy: f64) -> Result<(), ModelError> {
        if !energy.is_finite() {
            return Err(ModelError::NonFiniteValue {
                field: "onsite energy",
            });
        }
        let value = self
            .onsite
            .get_mut(orbital.index())
            .ok_or(ModelError::UnknownOrbital {
                index: orbital.index(),
            })?;
        *value = energy;
        Ok(())
    }

    /// Adds one hopping; the Hermitian-conjugate term is generated implicitly.
    pub fn add_hopping(
        &mut self,
        from: OrbitalId,
        to: OrbitalId,
        cell_offset: impl IntoIterator<Item = i32>,
        amplitude: Complex64,
    ) -> Result<(), ModelError> {
        self.validate_orbital(from)?;
        self.validate_orbital(to)?;

        let cell_offset: Vec<i32> = cell_offset.into_iter().collect();
        if cell_offset.len() != self.lattice.periodic_dimension() {
            return Err(ModelError::InvalidCellOffset {
                expected: self.lattice.periodic_dimension(),
                actual: cell_offset.len(),
            });
        }
        if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
            return Err(ModelError::NonFiniteValue {
                field: "hopping amplitude",
            });
        }
        if self
            .hoppings
            .iter()
            .any(|term| is_same_or_hermitian_partner(term, from, to, &cell_offset))
        {
            return Err(ModelError::DuplicateHopping);
        }

        self.hoppings.push(Hopping {
            from,
            to,
            cell_offset,
            amplitude,
        });
        Ok(())
    }

    /// Finalizes structural validation and returns an immutable model.
    pub fn build(self) -> Result<TightBindingModel, ModelError> {
        if self.orbitals.is_empty() {
            return Err(ModelError::EmptyModel);
        }
        Ok(TightBindingModel {
            lattice: self.lattice,
            orbitals: self.orbitals,
            onsite: self.onsite,
            hoppings: self.hoppings,
        })
    }

    fn validate_orbital(&self, orbital: OrbitalId) -> Result<(), ModelError> {
        if orbital.index() >= self.orbitals.len() {
            return Err(ModelError::UnknownOrbital {
                index: orbital.index(),
            });
        }
        Ok(())
    }
}

/// Immutable structural representation of a tight-binding model.
#[derive(Clone, Debug, PartialEq)]
pub struct TightBindingModel {
    lattice: Lattice,
    orbitals: Vec<Orbital>,
    onsite: Vec<f64>,
    hoppings: Vec<Hopping>,
}

impl TightBindingModel {
    /// Returns the translation structure.
    #[must_use]
    pub const fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    /// Returns all orbitals in stable identifier order.
    #[must_use]
    pub fn orbitals(&self) -> &[Orbital] {
        &self.orbitals
    }

    /// Returns the onsite energy for a valid model-local identifier.
    #[must_use]
    pub fn onsite(&self, orbital: OrbitalId) -> Option<f64> {
        self.onsite.get(orbital.index()).copied()
    }

    /// Returns all explicit hoppings. Hermitian partners are implicit.
    #[must_use]
    pub fn hoppings(&self) -> &[Hopping] {
        &self.hoppings
    }
}

fn is_same_or_hermitian_partner(
    existing: &Hopping,
    from: OrbitalId,
    to: OrbitalId,
    cell_offset: &[i32],
) -> bool {
    let same = existing.from == from
        && existing.to == to
        && existing.cell_offset == cell_offset;
    let partner = existing.from == to
        && existing.to == from
        && existing.cell_offset.len() == cell_offset.len()
        && existing
            .cell_offset
            .iter()
            .zip(cell_offset)
            .all(|(left, right)| i64::from(*left) == -i64::from(*right));
    same || partner
}
