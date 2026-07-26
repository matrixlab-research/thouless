//! Variable-orbital finite-system matrices assembled from site-local blocks.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::{Complex64, ComplexMatrix};

/// One directed Hamiltonian block between two finite-system sites.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteBlock {
    row_site: usize,
    column_site: usize,
    matrix: ComplexMatrix,
}

impl SiteBlock {
    /// Creates a directed site block.
    #[must_use]
    pub const fn new(row_site: usize, column_site: usize, matrix: ComplexMatrix) -> Self {
        Self {
            row_site,
            column_site,
            matrix,
        }
    }

    /// Site supplying the block rows.
    #[must_use]
    pub const fn row_site(&self) -> usize {
        self.row_site
    }

    /// Site supplying the block columns.
    #[must_use]
    pub const fn column_site(&self) -> usize {
        self.column_site
    }

    /// Dense local block.
    #[must_use]
    pub const fn matrix(&self) -> &ComplexMatrix {
        &self.matrix
    }
}

/// Canonical CSR storage for a selected block-system submatrix.
///
/// Unlike solver-specific CSR operators, a selected submatrix may have zero
/// rows, zero columns, or be rectangular.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockCsrMatrix {
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<Complex64>,
}

impl BlockCsrMatrix {
    /// Matrix shape.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    /// CSR row offsets.
    #[must_use]
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// Canonically sorted CSR column indices.
    #[must_use]
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    /// Stored nonzero values.
    #[must_use]
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Number of explicitly stored entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Errors raised by finite block-system assembly.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlockSystemError {
    /// Every site must contain at least one orbital.
    ZeroSiteDimension {
        /// Site with zero orbitals.
        site: usize,
    },
    /// A selected or block endpoint does not name a site.
    SiteOutOfBounds {
        /// Invalid site.
        site: usize,
        /// Number of sites.
        site_count: usize,
    },
    /// A block shape does not match its endpoint orbital counts.
    InvalidBlockShape {
        /// Block row site.
        row_site: usize,
        /// Block column site.
        column_site: usize,
        /// Required row count.
        expected_rows: usize,
        /// Required column count.
        expected_columns: usize,
        /// Supplied row count.
        actual_rows: usize,
        /// Supplied column count.
        actual_columns: usize,
    },
    /// A matrix dimension or sparse offset overflowed.
    DimensionOverflow,
}

impl fmt::Display for BlockSystemError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSiteDimension { site } => {
                write!(formatter, "site {site} has zero orbitals")
            }
            Self::SiteOutOfBounds { site, site_count } => {
                write!(
                    formatter,
                    "site {site} is outside a system with {site_count} sites"
                )
            }
            Self::InvalidBlockShape {
                row_site,
                column_site,
                expected_rows,
                expected_columns,
                actual_rows,
                actual_columns,
            } => write!(
                formatter,
                "block ({row_site}, {column_site}) has shape \
                 {actual_rows}x{actual_columns}; expected \
                 {expected_rows}x{expected_columns}"
            ),
            Self::DimensionOverflow => write!(formatter, "block-system dimension overflowed"),
        }
    }
}

impl Error for BlockSystemError {}

fn selection_offsets(site_dofs: &[usize], sites: &[usize]) -> Result<Vec<usize>, BlockSystemError> {
    let mut offsets = Vec::with_capacity(sites.len() + 1);
    offsets.push(0usize);
    for &site in sites {
        let dofs = *site_dofs
            .get(site)
            .ok_or(BlockSystemError::SiteOutOfBounds {
                site,
                site_count: site_dofs.len(),
            })?;
        let next = offsets
            .last()
            .copied()
            .expect("offset zero was inserted")
            .checked_add(dofs)
            .ok_or(BlockSystemError::DimensionOverflow)?;
        offsets.push(next);
    }
    Ok(offsets)
}

/// Assemble a selected finite-system Hamiltonian directly into canonical CSR.
///
/// `row_sites` and `column_sites` preserve caller order and may contain
/// repeated sites. Duplicate block contributions are summed. The work and
/// storage scale with selected block nonzeros rather than with the square of
/// the full orbital dimension.
pub fn assemble_block_csr(
    site_dofs: &[usize],
    blocks: &[SiteBlock],
    row_sites: &[usize],
    column_sites: &[usize],
) -> Result<BlockCsrMatrix, BlockSystemError> {
    for (site, &dofs) in site_dofs.iter().enumerate() {
        if dofs == 0 {
            return Err(BlockSystemError::ZeroSiteDimension { site });
        }
    }
    let row_basis_offsets = selection_offsets(site_dofs, row_sites)?;
    let column_basis_offsets = selection_offsets(site_dofs, column_sites)?;
    let row_count = *row_basis_offsets
        .last()
        .expect("selection offsets always contain zero");
    let column_count = *column_basis_offsets
        .last()
        .expect("selection offsets always contain zero");
    let mut row_instances = vec![Vec::new(); site_dofs.len()];
    for (position, &site) in row_sites.iter().enumerate() {
        row_instances[site].push(row_basis_offsets[position]);
    }
    let mut column_instances = vec![Vec::new(); site_dofs.len()];
    for (position, &site) in column_sites.iter().enumerate() {
        column_instances[site].push(column_basis_offsets[position]);
    }
    let mut assembled = vec![BTreeMap::<usize, Complex64>::new(); row_count];

    for block in blocks {
        let row_dofs = *site_dofs
            .get(block.row_site)
            .ok_or(BlockSystemError::SiteOutOfBounds {
                site: block.row_site,
                site_count: site_dofs.len(),
            })?;
        let column_dofs =
            *site_dofs
                .get(block.column_site)
                .ok_or(BlockSystemError::SiteOutOfBounds {
                    site: block.column_site,
                    site_count: site_dofs.len(),
                })?;
        if block.matrix.shape() != (row_dofs, column_dofs) {
            return Err(BlockSystemError::InvalidBlockShape {
                row_site: block.row_site,
                column_site: block.column_site,
                expected_rows: row_dofs,
                expected_columns: column_dofs,
                actual_rows: block.matrix.rows(),
                actual_columns: block.matrix.columns(),
            });
        }
        for &row_start in &row_instances[block.row_site] {
            for &column_start in &column_instances[block.column_site] {
                for local_row in 0..row_dofs {
                    for local_column in 0..column_dofs {
                        let value = block.matrix.as_slice()[local_row * column_dofs + local_column];
                        if value != Complex64::new(0.0, 0.0) {
                            *assembled[row_start + local_row]
                                .entry(column_start + local_column)
                                .or_default() += value;
                        }
                    }
                }
            }
        }
    }

    let mut row_offsets = Vec::with_capacity(row_count + 1);
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0);
    for row in assembled {
        for (column, value) in row {
            if value != Complex64::new(0.0, 0.0) {
                column_indices.push(column);
                values.push(value);
            }
        }
        row_offsets.push(values.len());
    }
    Ok(BlockCsrMatrix {
        rows: row_count,
        columns: column_count,
        row_offsets,
        column_indices,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(matrix: &BlockCsrMatrix) -> Vec<Complex64> {
        let (rows, columns) = matrix.shape();
        let mut result = vec![Complex64::new(0.0, 0.0); rows * columns];
        for row in 0..rows {
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                result[row * columns + matrix.column_indices()[entry]] = matrix.values()[entry];
            }
        }
        result
    }

    #[test]
    fn mixed_site_blocks_preserve_selection_order_and_repetition() {
        let blocks = vec![
            SiteBlock::new(
                0,
                0,
                ComplexMatrix::new(
                    2,
                    2,
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 1.0),
                        Complex64::new(0.0, -1.0),
                        Complex64::new(2.0, 0.0),
                    ],
                )
                .unwrap(),
            ),
            SiteBlock::new(1, 1, ComplexMatrix::scalar(Complex64::new(3.0, 0.0))),
            SiteBlock::new(
                0,
                1,
                ComplexMatrix::new(
                    2,
                    1,
                    vec![Complex64::new(4.0, 0.0), Complex64::new(5.0, 0.0)],
                )
                .unwrap(),
            ),
            SiteBlock::new(
                1,
                0,
                ComplexMatrix::new(
                    1,
                    2,
                    vec![Complex64::new(4.0, 0.0), Complex64::new(5.0, 0.0)],
                )
                .unwrap(),
            ),
        ];
        let matrix = assemble_block_csr(&[2, 1], &blocks, &[1, 0, 1], &[0, 1]).unwrap();
        assert_eq!(matrix.shape(), (4, 3));
        assert_eq!(
            dense(&matrix),
            vec![
                4.0.into(),
                5.0.into(),
                3.0.into(),
                1.0.into(),
                Complex64::new(0.0, 1.0),
                4.0.into(),
                Complex64::new(0.0, -1.0),
                2.0.into(),
                5.0.into(),
                4.0.into(),
                5.0.into(),
                3.0.into(),
            ]
        );
    }

    #[test]
    fn empty_and_rectangular_selections_are_explicit() {
        let empty = assemble_block_csr(&[1], &[], &[], &[0]).unwrap();
        assert_eq!(empty.shape(), (0, 1));
        assert_eq!(empty.row_offsets(), [0]);
        let rectangular = assemble_block_csr(&[2, 1], &[], &[0], &[1]).unwrap();
        assert_eq!(rectangular.shape(), (2, 1));
        assert_eq!(rectangular.row_offsets(), [0, 0, 0]);
    }

    #[test]
    fn duplicate_block_contributions_are_summed_canonically() {
        let blocks = [
            SiteBlock::new(0, 0, ComplexMatrix::scalar(Complex64::new(1.0, 0.5))),
            SiteBlock::new(0, 0, ComplexMatrix::scalar(Complex64::new(2.0, -0.5))),
        ];
        let matrix = assemble_block_csr(&[1], &blocks, &[0], &[0]).unwrap();
        assert_eq!(matrix.row_offsets(), [0, 1]);
        assert_eq!(matrix.column_indices(), [0]);
        assert_eq!(matrix.values(), [Complex64::new(3.0, 0.0)]);
    }

    #[test]
    fn invalid_block_shapes_are_rejected_before_assembly() {
        let error = assemble_block_csr(
            &[2, 1],
            &[SiteBlock::new(
                0,
                1,
                ComplexMatrix::scalar(Complex64::new(1.0, 0.0)),
            )],
            &[0],
            &[1],
        )
        .unwrap_err();
        assert_eq!(
            error,
            BlockSystemError::InvalidBlockShape {
                row_site: 0,
                column_site: 1,
                expected_rows: 2,
                expected_columns: 1,
                actual_rows: 1,
                actual_columns: 1,
            }
        );
    }
}
