use thouless::{Complex64, ComplexMatrix, MatrixError};

#[test]
fn adjoint_round_trip_and_hermiticity_are_consistent() {
    let matrix = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 3.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .expect("valid matrix");
    assert!(matrix.is_hermitian(1.0e-14).expect("square matrix"));
    assert_eq!(matrix.adjoint(), matrix);
}

#[test]
fn shape_and_bounds_fail_explicitly() {
    assert_eq!(
        ComplexMatrix::new(2, 2, vec![Complex64::new(0.0, 0.0)]).expect_err("wrong data length"),
        MatrixError::InvalidDataLength {
            rows: 2,
            columns: 2,
            actual: 1,
        }
    );
    let matrix = ComplexMatrix::zeros(2, 3);
    assert!(matches!(
        matrix.get(2, 0),
        Err(MatrixError::IndexOutOfBounds { .. })
    ));
}
