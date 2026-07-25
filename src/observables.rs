//! Local observables and projection into sampled state subspaces.

use crate::{Complex64, ComplexMatrix, ObservableError};

/// Contiguous basis-state blocks associated with physical sites.
///
/// The layout is backend independent. It records only the number of internal
/// degrees of freedom on each site and therefore supports scalar orbitals,
/// spinors, Nambu spaces, and heterogeneous site families.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalBasisLayout {
    dimensions: Vec<usize>,
    offsets: Vec<usize>,
}

impl LocalBasisLayout {
    /// Creates a layout from the number of basis states on each site.
    pub fn new(dimensions: impl IntoIterator<Item = usize>) -> Result<Self, ObservableError> {
        let dimensions: Vec<_> = dimensions.into_iter().collect();
        if dimensions.is_empty() {
            return Err(ObservableError::EmptyLocalBasis);
        }

        let mut offsets: Vec<usize> = Vec::with_capacity(dimensions.len() + 1);
        offsets.push(0);
        for (site, &dimension) in dimensions.iter().enumerate() {
            if dimension == 0 {
                return Err(ObservableError::EmptyLocalSite { site });
            }
            let next = offsets
                .last()
                .copied()
                .expect("the zero offset is present")
                .checked_add(dimension)
                .ok_or(ObservableError::LocalBasisSizeOverflow)?;
            offsets.push(next);
        }

        Ok(Self {
            dimensions,
            offsets,
        })
    }

    /// Returns the number of physical sites.
    #[must_use]
    pub fn site_count(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the total Hilbert-space dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        *self
            .offsets
            .last()
            .expect("a nonempty layout has a final offset")
    }

    /// Returns the number of basis states on one site.
    pub fn site_dimension(&self, site: usize) -> Result<usize, ObservableError> {
        self.dimensions
            .get(site)
            .copied()
            .ok_or(ObservableError::InvalidLocalSite {
                site,
                site_count: self.site_count(),
            })
    }

    fn offset(&self, site: usize) -> Result<usize, ObservableError> {
        if site >= self.site_count() {
            return Err(ObservableError::InvalidLocalSite {
                site,
                site_count: self.site_count(),
            });
        }
        Ok(self.offsets[site])
    }
}

/// One block in a sparse local operator.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalOperatorTerm {
    row_site: usize,
    column_site: usize,
    block: ComplexMatrix,
}

impl LocalOperatorTerm {
    /// Creates a block acting from `column_site` to `row_site`.
    #[must_use]
    pub fn new(row_site: usize, column_site: usize, block: ComplexMatrix) -> Self {
        Self {
            row_site,
            column_site,
            block,
        }
    }

    /// Returns the row-site index.
    #[must_use]
    pub const fn row_site(&self) -> usize {
        self.row_site
    }

    /// Returns the column-site index.
    #[must_use]
    pub const fn column_site(&self) -> usize {
        self.column_site
    }

    /// Returns the dense internal block.
    #[must_use]
    pub const fn block(&self) -> &ComplexMatrix {
        &self.block
    }
}

/// One independently resolved contribution to a local observable.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalOperatorComponent {
    terms: Vec<LocalOperatorTerm>,
}

impl LocalOperatorComponent {
    /// Creates a component from one or more sparse site-to-site blocks.
    #[must_use]
    pub fn new(terms: impl IntoIterator<Item = LocalOperatorTerm>) -> Self {
        Self {
            terms: terms.into_iter().collect(),
        }
    }

    /// Returns the sparse blocks in this component.
    #[must_use]
    pub fn terms(&self) -> &[LocalOperatorTerm] {
        &self.terms
    }
}

/// An additive collection of block-sparse local observable components.
///
/// Components remain resolved for site- or bond-level analysis. The total
/// operator can be applied or evaluated without materializing dense matrices;
/// dense conversion is available only at an explicit interoperability
/// boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalOperatorSet {
    layout: LocalBasisLayout,
    components: Vec<LocalOperatorComponent>,
}

impl LocalOperatorSet {
    /// Creates and validates a resolved local operator.
    pub fn new(
        layout: LocalBasisLayout,
        components: impl IntoIterator<Item = LocalOperatorComponent>,
    ) -> Result<Self, ObservableError> {
        let components: Vec<_> = components.into_iter().collect();
        for component in &components {
            for term in component.terms() {
                validate_term(&layout, term)?;
            }
        }
        Ok(Self { layout, components })
    }

    /// Returns the local basis layout.
    #[must_use]
    pub const fn layout(&self) -> &LocalBasisLayout {
        &self.layout
    }

    /// Returns the resolved components.
    #[must_use]
    pub fn components(&self) -> &[LocalOperatorComponent] {
        &self.components
    }

    /// Evaluates `<bra|O_c|ket>` for every resolved component.
    pub fn matrix_elements(
        &self,
        bra: &[Complex64],
        ket: &[Complex64],
    ) -> Result<Vec<Complex64>, ObservableError> {
        validate_state(&self.layout, bra)?;
        validate_state(&self.layout, ket)?;
        let values: Vec<_> = self
            .components
            .iter()
            .map(|component| component_matrix_element(&self.layout, component, bra, ket))
            .collect();
        if values
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(ObservableError::NonFiniteValue);
        }
        Ok(values)
    }

    /// Applies the sum of all resolved components to a state.
    pub fn apply_total(&self, ket: &[Complex64]) -> Result<Vec<Complex64>, ObservableError> {
        validate_state(&self.layout, ket)?;
        let mut result = vec![Complex64::new(0.0, 0.0); self.layout.dimension()];
        for component in &self.components {
            for term in component.terms() {
                let row_offset = self
                    .layout
                    .offset(term.row_site())
                    .expect("validated row site");
                let column_offset = self
                    .layout
                    .offset(term.column_site())
                    .expect("validated column site");
                for row in 0..term.block().rows() {
                    for column in 0..term.block().columns() {
                        result[row_offset + row] += term.block().as_slice()
                            [row * term.block().columns() + column]
                            * ket[column_offset + column];
                    }
                }
            }
        }
        if result
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(ObservableError::NonFiniteValue);
        }
        Ok(result)
    }

    /// Materializes every resolved component as a dense matrix.
    pub fn component_matrices(&self) -> Result<Vec<ComplexMatrix>, ObservableError> {
        self.components
            .iter()
            .map(|component| materialize_component(&self.layout, component))
            .collect::<Result<Vec<_>, _>>()
    }

    /// Materializes the sum of all resolved components as a dense matrix.
    pub fn total_matrix(&self) -> Result<ComplexMatrix, ObservableError> {
        let dimension = self.layout.dimension();
        let entry_count = dimension
            .checked_mul(dimension)
            .ok_or(ObservableError::DenseLocalOperatorSizeOverflow { dimension })?;
        let mut data = vec![Complex64::new(0.0, 0.0); entry_count];
        for component in &self.components {
            add_component_to_dense(&self.layout, component, &mut data);
        }
        ComplexMatrix::new(dimension, dimension, data).map_err(|_| ObservableError::NonFiniteValue)
    }
}

/// One site-resolved density block.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalDensityTerm {
    site: usize,
    observable: ComplexMatrix,
}

impl LocalDensityTerm {
    /// Creates a site-resolved density `Q_i`.
    #[must_use]
    pub fn new(site: usize, observable: ComplexMatrix) -> Self {
        Self { site, observable }
    }
}

/// One oriented bond contribution to a local continuity equation.
#[derive(Clone, Debug, PartialEq)]
pub struct BondCurrentTerm {
    site: usize,
    neighbor: usize,
    observable: ComplexMatrix,
    hopping: ComplexMatrix,
}

impl BondCurrentTerm {
    /// Creates the contribution of `H_(site,neighbor)` to `d<Q_site>/dt`.
    ///
    /// The returned current uses units with `ℏ = 1`. `observable` acts on
    /// `site`, and `hopping` maps the neighbor basis into the site basis.
    #[must_use]
    pub fn new(
        site: usize,
        neighbor: usize,
        observable: ComplexMatrix,
        hopping: ComplexMatrix,
    ) -> Self {
        Self {
            site,
            neighbor,
            observable,
            hopping,
        }
    }
}

/// One onsite production term in a local continuity equation.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalSourceTerm {
    site: usize,
    observable: ComplexMatrix,
    onsite: ComplexMatrix,
}

impl LocalSourceTerm {
    /// Creates the onsite contribution `i(H_ii Q_i - Q_i H_ii)`.
    ///
    /// The returned source uses units with `ℏ = 1`.
    #[must_use]
    pub fn new(site: usize, observable: ComplexMatrix, onsite: ComplexMatrix) -> Self {
        Self {
            site,
            observable,
            onsite,
        }
    }
}

/// Resolves site-local density operators.
pub fn local_densities(
    layout: &LocalBasisLayout,
    densities: &[LocalDensityTerm],
) -> Result<LocalOperatorSet, ObservableError> {
    LocalOperatorSet::new(
        layout.clone(),
        densities.iter().map(|density| {
            LocalOperatorComponent::new([LocalOperatorTerm::new(
                density.site,
                density.site,
                density.observable.clone(),
            )])
        }),
    )
}

/// Resolves oriented bond-current operators in a local continuity equation.
///
/// For a local observable `Q_i` and hopping `H_ij`, the two nonzero blocks are
/// `J_ij[i,j] = -i Q_i H_ij` and
/// `J_ij[j,i] = i H_ij† Q_i`. Thus `J_ij` is precisely the bond contribution
/// to the Heisenberg rate `i[H, Q_i]` in units with `ℏ = 1`.
pub fn bond_currents(
    layout: &LocalBasisLayout,
    currents: &[BondCurrentTerm],
) -> Result<LocalOperatorSet, ObservableError> {
    let mut components = Vec::with_capacity(currents.len());
    for current in currents {
        validate_square_site_matrix(layout, current.site, &current.observable)?;
        validate_site_block(layout, current.site, current.neighbor, &current.hopping)?;
        let forward = scale_matrix(
            &multiply(&current.observable, &current.hopping)?,
            Complex64::new(0.0, -1.0),
        )?;
        let reverse = scale_matrix(
            &multiply(&current.hopping.adjoint(), &current.observable)?,
            Complex64::new(0.0, 1.0),
        )?;
        components.push(LocalOperatorComponent::new([
            LocalOperatorTerm::new(current.site, current.neighbor, forward),
            LocalOperatorTerm::new(current.neighbor, current.site, reverse),
        ]));
    }
    LocalOperatorSet::new(layout.clone(), components)
}

/// Resolves onsite production operators in a local continuity equation.
///
/// Each source is `i(H_ii Q_i - Q_i H_ii)` in units with `ℏ = 1`.
pub fn local_sources(
    layout: &LocalBasisLayout,
    sources: &[LocalSourceTerm],
) -> Result<LocalOperatorSet, ObservableError> {
    let mut components = Vec::with_capacity(sources.len());
    for source in sources {
        validate_square_site_matrix(layout, source.site, &source.observable)?;
        validate_square_site_matrix(layout, source.site, &source.onsite)?;
        let hq = multiply(&source.onsite, &source.observable)?;
        let qh = multiply(&source.observable, &source.onsite)?;
        let block = ComplexMatrix::new(
            hq.rows(),
            hq.columns(),
            hq.as_slice()
                .iter()
                .zip(qh.as_slice())
                .map(|(&left, &right)| Complex64::new(0.0, 1.0) * (left - right))
                .collect(),
        )
        .map_err(|_| ObservableError::NonFiniteValue)?;
        components.push(LocalOperatorComponent::new([LocalOperatorTerm::new(
            source.site,
            source.site,
            block,
        )]));
    }
    LocalOperatorSet::new(layout.clone(), components)
}

/// Projects a real diagonal basis observable into a state subspace.
///
/// State vectors are rows of `states`; `diagonal[b]` is the observable value
/// of basis state `b`. The result is the Hermitian matrix
/// `O_mn = Σ_b ψ*_mb O_b ψ_nb`.
pub fn project_diagonal_observable(
    states: &ComplexMatrix,
    diagonal: &[f64],
) -> Result<ComplexMatrix, ObservableError> {
    if states.rows() == 0 || states.columns() == 0 {
        return Err(ObservableError::EmptyStateFrame);
    }
    if diagonal.len() != states.columns() {
        return Err(ObservableError::InvalidDiagonalLength {
            expected: states.columns(),
            actual: diagonal.len(),
        });
    }
    if diagonal.iter().any(|value| !value.is_finite()) {
        return Err(ObservableError::NonFiniteValue);
    }

    let state_count = states.rows();
    let basis_count = states.columns();
    let mut projected = ComplexMatrix::zeros(state_count, state_count);
    for bra in 0..state_count {
        for ket in 0..state_count {
            let value: Complex64 = (0..basis_count)
                .map(|basis| {
                    states.as_slice()[bra * basis_count + basis].conj()
                        * diagonal[basis]
                        * states.as_slice()[ket * basis_count + basis]
                })
                .sum();
            projected
                .set(bra, ket, value)
                .expect("projected indices are in bounds");
        }
    }
    Ok(projected)
}

/// Decomposes a two-state operator into identity and Pauli coefficients.
pub fn pauli_coefficients(matrix: &ComplexMatrix) -> Result<[Complex64; 4], ObservableError> {
    if matrix.shape() != (2, 2) {
        return Err(ObservableError::InvalidPauliShape {
            rows: matrix.rows(),
            columns: matrix.columns(),
        });
    }
    let half = Complex64::new(0.5, 0.0);
    let imaginary_half = Complex64::new(0.0, 0.5);
    let m00 = matrix.as_slice()[0];
    let m01 = matrix.as_slice()[1];
    let m10 = matrix.as_slice()[2];
    let m11 = matrix.as_slice()[3];
    Ok([
        half * (m00 + m11),
        half * (m01 + m10),
        imaginary_half * (m01 - m10),
        half * (m00 - m11),
    ])
}

fn validate_state(layout: &LocalBasisLayout, state: &[Complex64]) -> Result<(), ObservableError> {
    if state.len() != layout.dimension() {
        return Err(ObservableError::InvalidStateLength {
            expected: layout.dimension(),
            actual: state.len(),
        });
    }
    if state
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(ObservableError::NonFiniteStateValue);
    }
    Ok(())
}

fn validate_term(
    layout: &LocalBasisLayout,
    term: &LocalOperatorTerm,
) -> Result<(), ObservableError> {
    validate_site_block(layout, term.row_site(), term.column_site(), term.block())
}

fn validate_square_site_matrix(
    layout: &LocalBasisLayout,
    site: usize,
    matrix: &ComplexMatrix,
) -> Result<(), ObservableError> {
    validate_site_block(layout, site, site, matrix)
}

fn validate_site_block(
    layout: &LocalBasisLayout,
    row_site: usize,
    column_site: usize,
    block: &ComplexMatrix,
) -> Result<(), ObservableError> {
    let expected_rows = layout.site_dimension(row_site)?;
    let expected_columns = layout.site_dimension(column_site)?;
    if block.shape() != (expected_rows, expected_columns) {
        return Err(ObservableError::InvalidLocalBlockShape {
            row_site,
            column_site,
            expected_rows,
            expected_columns,
            actual_rows: block.rows(),
            actual_columns: block.columns(),
        });
    }
    Ok(())
}

fn component_matrix_element(
    layout: &LocalBasisLayout,
    component: &LocalOperatorComponent,
    bra: &[Complex64],
    ket: &[Complex64],
) -> Complex64 {
    component
        .terms()
        .iter()
        .map(|term| {
            let row_offset = layout.offset(term.row_site()).expect("validated row site");
            let column_offset = layout
                .offset(term.column_site())
                .expect("validated column site");
            (0..term.block().rows())
                .flat_map(|row| {
                    (0..term.block().columns()).map(move |column| {
                        bra[row_offset + row].conj()
                            * term.block().as_slice()[row * term.block().columns() + column]
                            * ket[column_offset + column]
                    })
                })
                .sum::<Complex64>()
        })
        .sum()
}

fn materialize_component(
    layout: &LocalBasisLayout,
    component: &LocalOperatorComponent,
) -> Result<ComplexMatrix, ObservableError> {
    let dimension = layout.dimension();
    let entry_count = dimension
        .checked_mul(dimension)
        .ok_or(ObservableError::DenseLocalOperatorSizeOverflow { dimension })?;
    let mut data = vec![Complex64::new(0.0, 0.0); entry_count];
    add_component_to_dense(layout, component, &mut data);
    ComplexMatrix::new(dimension, dimension, data).map_err(|_| ObservableError::NonFiniteValue)
}

fn add_component_to_dense(
    layout: &LocalBasisLayout,
    component: &LocalOperatorComponent,
    data: &mut [Complex64],
) {
    let dimension = layout.dimension();
    for term in component.terms() {
        let row_offset = layout.offset(term.row_site()).expect("validated row site");
        let column_offset = layout
            .offset(term.column_site())
            .expect("validated column site");
        for row in 0..term.block().rows() {
            for column in 0..term.block().columns() {
                data[(row_offset + row) * dimension + column_offset + column] +=
                    term.block().as_slice()[row * term.block().columns() + column];
            }
        }
    }
}

fn multiply(left: &ComplexMatrix, right: &ComplexMatrix) -> Result<ComplexMatrix, ObservableError> {
    debug_assert_eq!(left.columns(), right.rows());
    let mut data = vec![Complex64::new(0.0, 0.0); left.rows() * right.columns()];
    for row in 0..left.rows() {
        for column in 0..right.columns() {
            data[row * right.columns() + column] = (0..left.columns())
                .map(|inner| {
                    left.as_slice()[row * left.columns() + inner]
                        * right.as_slice()[inner * right.columns() + column]
                })
                .sum();
        }
    }
    ComplexMatrix::new(left.rows(), right.columns(), data)
        .map_err(|_| ObservableError::NonFiniteValue)
}

fn scale_matrix(
    matrix: &ComplexMatrix,
    factor: Complex64,
) -> Result<ComplexMatrix, ObservableError> {
    ComplexMatrix::new(
        matrix.rows(),
        matrix.columns(),
        matrix
            .as_slice()
            .iter()
            .map(|&value| factor * value)
            .collect(),
    )
    .map_err(|_| ObservableError::NonFiniteValue)
}
