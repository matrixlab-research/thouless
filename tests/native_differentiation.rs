use thouless::differentiation::{finite_difference_uniform, DifferenceScheme};
use thouless::model::{Lattice, ModelBuilder};
use thouless::{Complex64, ComplexMatrix};

#[test]
fn finite_difference_recovers_linear_matrix_family() {
    let samples: Vec<_> = [0.0, 0.25, 0.5]
        .into_iter()
        .map(|value| ComplexMatrix::scalar(Complex64::new(2.0 * value, 0.0)))
        .collect();
    for scheme in [DifferenceScheme::Central, DifferenceScheme::Forward] {
        let derivatives = finite_difference_uniform(&samples, 0.25, false, scheme).unwrap();
        for derivative in derivatives {
            assert_eq!(derivative.get(0, 0).unwrap(), Complex64::new(2.0, 0.0));
        }
    }
}

#[test]
fn cosine_chain_derivative_is_analytic_and_hermitian() {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder.add_orbital("s", [0.0]).unwrap();
    builder
        .add_hopping(orbital, orbital, [1], Complex64::new(-1.0, 0.0))
        .unwrap();
    let model = builder.build().unwrap();

    let momentum = 0.25;
    let derivative = &model.reduced_momentum_derivatives(&[momentum]).unwrap()[0];
    let expected = 4.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * momentum).sin();
    assert!((derivative.get(0, 0).unwrap().re - expected).abs() < 1.0e-12);
    assert!(derivative.is_hermitian(0.0).unwrap());
}
