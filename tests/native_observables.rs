use thouless::observables::{pauli_coefficients, project_diagonal_observable};
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
