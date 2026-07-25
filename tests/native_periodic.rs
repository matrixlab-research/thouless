use std::f64::consts::TAU;

use thouless::periodic::{fold_terms, PeriodicTerm};
use thouless::{Complex64, ComplexMatrix};

#[test]
fn integer_translation_folding_is_reciprocal_periodic() {
    let hopping = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.2, 0.1),
            Complex64::new(-0.3, 0.4),
            Complex64::new(0.7, -0.2),
            Complex64::new(-0.1, 0.5),
        ],
    )
    .unwrap();
    let terms = [PeriodicTerm::new(hopping, [3, -2], false)];
    let first = fold_terms(&terms, &[0.31, -0.27]).unwrap();
    let shifted = fold_terms(&terms, &[0.31 + TAU, -0.27 - 2.0 * TAU]).unwrap();
    for (left, right) in first.as_slice().iter().zip(shifted.as_slice()) {
        assert!((*left - *right).norm() < 1.0e-13);
    }
}

#[test]
fn scalar_chain_folding_recovers_cosine_dispersion() {
    let hopping = PeriodicTerm::new(ComplexMatrix::scalar(Complex64::new(-1.3, 0.0)), [1], true);
    for momentum in [-2.7_f64, -0.4, 0.0, 1.2, 3.0] {
        let folded = fold_terms(std::slice::from_ref(&hopping), &[momentum]).unwrap();
        let expected = -2.6 * momentum.cos();
        assert!((folded.as_slice()[0].re - expected).abs() < 1.0e-13);
        assert!(folded.as_slice()[0].im.abs() < 1.0e-13);
    }
}
