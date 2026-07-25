use thouless::observables::{
    bond_currents, local_densities, local_sources, pauli_coefficients, project_diagonal_observable,
    BondCurrentTerm, LocalBasisLayout, LocalDensityTerm, LocalSourceTerm,
};
use thouless::ObservableError;
use thouless::{Complex64, ComplexMatrix};

#[test]
fn diagonal_observable_projects_into_a_rotated_subspace() {
    let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
    let states = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(-inverse_sqrt_two, 0.0),
        ],
    )
    .unwrap();

    let projected = project_diagonal_observable(&states, &[0.0, 2.0]).unwrap();
    assert!((projected.get(0, 0).unwrap() - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
    assert!((projected.get(1, 1).unwrap() - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
    assert!((projected.get(0, 1).unwrap() - Complex64::new(-1.0, 0.0)).norm() < 1.0e-12);
    assert!(projected.is_hermitian(1.0e-12).unwrap());
}

#[test]
fn pauli_decomposition_recovers_all_four_components() {
    let matrix = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(2.0, 3.0),
            Complex64::new(-3.0, 0.0),
        ],
    )
    .unwrap();
    let coefficients = pauli_coefficients(&matrix).unwrap();
    assert_eq!(
        coefficients,
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]
    );
}

#[test]
fn mixed_site_density_is_evaluated_without_dense_materialization() {
    let layout = LocalBasisLayout::new([1, 2]).unwrap();
    let density = local_densities(
        &layout,
        &[
            LocalDensityTerm::new(0, matrix(1, 1, &[(2.0, 0.0)])),
            LocalDensityTerm::new(
                1,
                matrix(2, 2, &[(1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, 0.0)]),
            ),
        ],
    )
    .unwrap();
    let bra = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(-0.2, 0.3),
        Complex64::new(0.7, 0.1),
    ];
    let ket = vec![
        Complex64::new(0.4, 0.2),
        Complex64::new(0.8, -0.1),
        Complex64::new(-0.3, 0.6),
    ];

    assert_eq!(density.components().len(), 2);
    assert_eq!(density.components()[0].terms().len(), 1);
    let values = density.matrix_elements(&bra, &ket).unwrap();
    let dense_components = density.component_matrices().unwrap();
    for (value, dense) in values.iter().zip(&dense_components) {
        assert!((*value - dense_matrix_element(&bra, dense, &ket)).norm() < 1.0e-12);
    }

    let applied = density.apply_total(&ket).unwrap();
    let expected = apply_dense(&density.total_matrix().unwrap(), &ket);
    assert_vectors_close(&applied, &expected, 1.0e-12);
}

#[test]
fn bond_and_source_terms_reconstruct_the_local_continuity_equation() {
    let layout = LocalBasisLayout::new([2, 1]).unwrap();
    let onsite = matrix(2, 2, &[(0.4, 0.0), (0.2, -0.3), (0.2, 0.3), (-0.1, 0.0)]);
    let neighbor_onsite = matrix(1, 1, &[(0.7, 0.0)]);
    let hopping = matrix(2, 1, &[(0.5, 0.1), (-0.2, 0.4)]);
    let observable = matrix(2, 2, &[(1.0, 0.0), (0.1, -0.2), (0.1, 0.2), (-0.4, 0.0)]);

    let current = bond_currents(
        &layout,
        &[BondCurrentTerm::new(
            0,
            1,
            observable.clone(),
            hopping.clone(),
        )],
    )
    .unwrap();
    let source = local_sources(
        &layout,
        &[LocalSourceTerm::new(0, observable.clone(), onsite.clone())],
    )
    .unwrap();
    assert_eq!(current.components()[0].terms().len(), 2);

    let hamiltonian = block_hamiltonian(&onsite, &neighbor_onsite, &hopping);
    let local_density = local_densities(&layout, &[LocalDensityTerm::new(0, observable)])
        .unwrap()
        .total_matrix()
        .unwrap();
    let expected_rate = scaled_difference(
        &multiply_dense(&hamiltonian, &local_density),
        &multiply_dense(&local_density, &hamiltonian),
        Complex64::new(0.0, 1.0),
    );
    let actual_rate = add_dense(
        &source.total_matrix().unwrap(),
        &current.total_matrix().unwrap(),
    );
    assert_matrices_close(&actual_rate, &expected_rate, 1.0e-12);
    assert!(actual_rate.is_hermitian(1.0e-12).unwrap());
}

#[test]
fn conserved_density_currents_are_antisymmetric_across_a_bond() {
    let layout = LocalBasisLayout::new([1, 2]).unwrap();
    let hopping = matrix(1, 2, &[(0.4, -0.2), (-0.1, 0.7)]);
    let forward = bond_currents(
        &layout,
        &[BondCurrentTerm::new(
            0,
            1,
            ComplexMatrix::identity(1),
            hopping.clone(),
        )],
    )
    .unwrap()
    .total_matrix()
    .unwrap();
    let reverse = bond_currents(
        &layout,
        &[BondCurrentTerm::new(
            1,
            0,
            ComplexMatrix::identity(2),
            hopping.adjoint(),
        )],
    )
    .unwrap()
    .total_matrix()
    .unwrap();
    let cancellation = add_dense(&forward, &reverse);
    assert!(cancellation
        .as_slice()
        .iter()
        .all(|value| value.norm() < 1.0e-12));
}

#[test]
fn bond_current_is_covariant_under_independent_site_gauges() {
    let layout = LocalBasisLayout::new([1, 1]).unwrap();
    let hopping = Complex64::new(0.6, -0.35);
    let state = [Complex64::new(0.7, 0.2), Complex64::new(-0.1, 0.65)];
    let current = bond_currents(
        &layout,
        &[BondCurrentTerm::new(
            0,
            1,
            ComplexMatrix::identity(1),
            ComplexMatrix::scalar(hopping),
        )],
    )
    .unwrap();
    let reference = current.matrix_elements(&state, &state).unwrap()[0];

    let first_gauge = Complex64::from_polar(1.0, 0.37);
    let second_gauge = Complex64::from_polar(1.0, -0.81);
    let transformed_state = [first_gauge * state[0], second_gauge * state[1]];
    let transformed_hopping = first_gauge * hopping * second_gauge.conj();
    let transformed = bond_currents(
        &layout,
        &[BondCurrentTerm::new(
            0,
            1,
            ComplexMatrix::identity(1),
            ComplexMatrix::scalar(transformed_hopping),
        )],
    )
    .unwrap()
    .matrix_elements(&transformed_state, &transformed_state)
    .unwrap()[0];

    assert!((transformed - reference).norm() < 1.0e-12);
}

#[test]
fn local_layout_and_operator_shapes_are_validated() {
    assert_eq!(
        LocalBasisLayout::new([]),
        Err(ObservableError::EmptyLocalBasis)
    );
    assert_eq!(
        LocalBasisLayout::new([1, 0]),
        Err(ObservableError::EmptyLocalSite { site: 1 })
    );
    let layout = LocalBasisLayout::new([1, 2]).unwrap();
    let error = local_densities(
        &layout,
        &[LocalDensityTerm::new(1, ComplexMatrix::identity(1))],
    )
    .unwrap_err();
    assert_eq!(
        error,
        ObservableError::InvalidLocalBlockShape {
            row_site: 1,
            column_site: 1,
            expected_rows: 2,
            expected_columns: 2,
            actual_rows: 1,
            actual_columns: 1,
        }
    );

    let empty = local_densities(&layout, &[]).unwrap();
    assert_eq!(empty.components().len(), 0);
    assert!(empty
        .total_matrix()
        .unwrap()
        .as_slice()
        .iter()
        .all(|value| value.norm() == 0.0));

    let huge_layout = LocalBasisLayout::new([usize::MAX]).unwrap();
    let huge = local_densities(&huge_layout, &[]).unwrap();
    assert_eq!(
        huge.total_matrix(),
        Err(ObservableError::DenseLocalOperatorSizeOverflow {
            dimension: usize::MAX,
        })
    );

    let scalar_layout = LocalBasisLayout::new([1, 1]).unwrap();
    assert_eq!(
        bond_currents(
            &scalar_layout,
            &[BondCurrentTerm::new(
                0,
                1,
                ComplexMatrix::scalar(Complex64::new(1.0e308, 0.0)),
                ComplexMatrix::scalar(Complex64::new(1.0e308, 0.0)),
            )],
        ),
        Err(ObservableError::NonFiniteValue)
    );
}

fn matrix(rows: usize, columns: usize, values: &[(f64, f64)]) -> ComplexMatrix {
    ComplexMatrix::new(
        rows,
        columns,
        values
            .iter()
            .map(|&(real, imaginary)| Complex64::new(real, imaginary))
            .collect(),
    )
    .unwrap()
}

fn dense_matrix_element(
    bra: &[Complex64],
    operator: &ComplexMatrix,
    ket: &[Complex64],
) -> Complex64 {
    bra.iter()
        .enumerate()
        .flat_map(|(row, &bra_value)| {
            ket.iter().enumerate().map(move |(column, &ket_value)| {
                bra_value.conj()
                    * operator.as_slice()[row * operator.columns() + column]
                    * ket_value
            })
        })
        .sum()
}

fn apply_dense(operator: &ComplexMatrix, ket: &[Complex64]) -> Vec<Complex64> {
    (0..operator.rows())
        .map(|row| {
            (0..operator.columns())
                .map(|column| operator.as_slice()[row * operator.columns() + column] * ket[column])
                .sum()
        })
        .collect()
}

fn multiply_dense(left: &ComplexMatrix, right: &ComplexMatrix) -> ComplexMatrix {
    assert_eq!(left.columns(), right.rows());
    ComplexMatrix::new(
        left.rows(),
        right.columns(),
        (0..left.rows())
            .flat_map(|row| {
                (0..right.columns()).map(move |column| {
                    (0..left.columns())
                        .map(|inner| {
                            left.as_slice()[row * left.columns() + inner]
                                * right.as_slice()[inner * right.columns() + column]
                        })
                        .sum()
                })
            })
            .collect(),
    )
    .unwrap()
}

fn add_dense(left: &ComplexMatrix, right: &ComplexMatrix) -> ComplexMatrix {
    assert_eq!(left.shape(), right.shape());
    ComplexMatrix::new(
        left.rows(),
        left.columns(),
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(&left, &right)| left + right)
            .collect(),
    )
    .unwrap()
}

fn scaled_difference(
    left: &ComplexMatrix,
    right: &ComplexMatrix,
    factor: Complex64,
) -> ComplexMatrix {
    assert_eq!(left.shape(), right.shape());
    ComplexMatrix::new(
        left.rows(),
        left.columns(),
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(&left, &right)| factor * (left - right))
            .collect(),
    )
    .unwrap()
}

fn block_hamiltonian(
    onsite: &ComplexMatrix,
    neighbor_onsite: &ComplexMatrix,
    hopping: &ComplexMatrix,
) -> ComplexMatrix {
    let mut result = ComplexMatrix::zeros(3, 3);
    for row in 0..2 {
        for column in 0..2 {
            result
                .set(row, column, onsite.get(row, column).unwrap())
                .unwrap();
        }
        result.set(row, 2, hopping.get(row, 0).unwrap()).unwrap();
        result
            .set(2, row, hopping.get(row, 0).unwrap().conj())
            .unwrap();
    }
    result
        .set(2, 2, neighbor_onsite.get(0, 0).unwrap())
        .unwrap();
    result
}

fn assert_vectors_close(left: &[Complex64], right: &[Complex64], tolerance: f64) {
    assert_eq!(left.len(), right.len());
    for (&left, &right) in left.iter().zip(right) {
        assert!((left - right).norm() < tolerance);
    }
}

fn assert_matrices_close(left: &ComplexMatrix, right: &ComplexMatrix, tolerance: f64) {
    assert_eq!(left.shape(), right.shape());
    for (&left, &right) in left.as_slice().iter().zip(right.as_slice()) {
        assert!((left - right).norm() < tolerance);
    }
}
