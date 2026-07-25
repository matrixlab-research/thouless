use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use thouless::matrix::ComplexMatrix;
use thouless::model::{Lattice, ModelBuilder, TightBindingModel};
use thouless::{Complex64, ModelError};

fn assert_close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "{left} differs from {right} by more than {tolerance}"
    );
}

#[test]
fn lattice_separates_real_dimension_from_periodic_axes() {
    let slab = Lattice::new(
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        vec![0, 2],
    )
    .expect("valid mixed geometry");
    assert_eq!(slab.real_dimension(), 3);
    assert_eq!(slab.periodic_dimension(), 2);
    assert_eq!(slab.periodic_axes(), &[0, 2]);

    let molecule = Lattice::new(Vec::new(), Vec::new()).expect("valid zero-dimensional model");
    assert_eq!(molecule.real_dimension(), 0);
    assert_eq!(molecule.periodic_dimension(), 0);
}

#[test]
fn scalar_chain_has_the_expected_cosine_dispersion() {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).expect("valid chain lattice");
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder.add_orbital("s", [0.0]).expect("valid orbital");
    builder.set_onsite(orbital, 0.25).expect("valid onsite");
    builder
        .add_hopping(orbital, orbital, [1], Complex64::new(-1.0, 0.0))
        .expect("valid hopping");
    let model = builder.build().expect("valid model");

    let gamma = model.eigensystem(&[0.0]).expect("solvable Hamiltonian");
    let zone_edge = model.eigensystem(&[0.5]).expect("solvable Hamiltonian");
    assert_close(gamma.eigenvalues()[0], -1.75, 1.0e-12);
    assert_close(zone_edge.eigenvalues()[0], 2.25, 1.0e-12);
}

#[test]
fn finite_dimer_has_bonding_and_antibonding_states() {
    let lattice = Lattice::new(Vec::new(), Vec::new()).expect("valid finite lattice");
    let mut builder = ModelBuilder::new(lattice);
    let left = builder
        .add_orbital("left", std::iter::empty())
        .expect("valid orbital");
    let right = builder
        .add_orbital("right", std::iter::empty())
        .expect("valid orbital");
    builder
        .add_hopping(left, right, std::iter::empty(), Complex64::new(-1.0, 0.0))
        .expect("valid hopping");
    let model = builder.build().expect("valid model");

    let spectrum = model.eigensystem(&[]).expect("solvable Hamiltonian");
    assert_close(spectrum.eigenvalues()[0], -1.0, 1.0e-12);
    assert_close(spectrum.eigenvalues()[1], 1.0, 1.0e-12);
}

#[test]
fn block_orbitals_support_spin_and_general_norbs() {
    let lattice = Lattice::new(Vec::new(), Vec::new()).expect("valid finite lattice");
    let mut builder = ModelBuilder::new(lattice);
    let spin = builder
        .add_orbital_with_dof("spinful", std::iter::empty(), 2)
        .expect("valid spinful orbital");
    let sigma_z = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-0.5, 0.0),
        ],
    )
    .expect("valid block");
    builder
        .set_onsite_block(spin, sigma_z)
        .expect("Hermitian onsite");
    let model = builder.build().expect("valid model");

    assert_eq!(model.state_count(), 2);
    let spectrum = model.eigensystem(&[]).expect("solvable Hamiltonian");
    assert_close(spectrum.eigenvalues()[0], -0.5, 1.0e-12);
    assert_close(spectrum.eigenvalues()[1], 0.5, 1.0e-12);
}

#[test]
fn orbital_embedding_changes_gauge_but_not_two_band_spectrum() {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).expect("valid chain lattice");
    let mut builder = ModelBuilder::new(lattice);
    let a = builder.add_orbital("a", [0.0]).expect("valid orbital");
    let b = builder.add_orbital("b", [0.5]).expect("valid orbital");
    builder
        .add_hopping(a, b, [0], Complex64::new(1.0, 0.0))
        .expect("intracell hopping");
    builder
        .add_hopping(a, b, [1], Complex64::new(0.6, 0.0))
        .expect("intercell hopping");
    let model = builder.build().expect("valid model");

    for momentum in [0.0, 0.125, 0.31, 0.5] {
        let spectrum = model
            .eigensystem(&[momentum])
            .expect("solvable Hamiltonian");
        let expected = (1.0_f64 + 0.36 + 1.2 * (2.0 * PI * momentum).cos()).sqrt();
        assert_close(spectrum.eigenvalues()[0], -expected, 1.0e-11);
        assert_close(spectrum.eigenvalues()[1], expected, 1.0e-11);
    }
}

#[test]
fn invalid_lattice_blocks_and_duplicate_hoppings_are_rejected() {
    assert!(matches!(
        Lattice::new(vec![vec![1.0, 0.0]], Vec::new()),
        Err(ModelError::InvalidPrimitiveVectors { .. })
    ));
    assert_eq!(
        Lattice::new(vec![vec![0.0]], Vec::new()).expect_err("singular lattice"),
        ModelError::SingularLattice
    );

    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).expect("valid chain lattice");
    let mut builder = ModelBuilder::new(lattice);
    let left = builder.add_orbital("left", [0.0]).expect("valid orbital");
    let right = builder.add_orbital("right", [0.5]).expect("valid orbital");
    builder
        .add_hopping(left, right, [1], Complex64::new(-1.0, 0.25))
        .expect("first hopping is valid");
    let error = builder
        .add_hopping(right, left, [-1], Complex64::new(-1.0, -0.25))
        .expect_err("reverse term must not be stored twice");
    assert_eq!(error, ModelError::DuplicateHopping);
}

struct DeterministicValues {
    state: u64,
}

impl DeterministicValues {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 11) as f64) / ((1_u64 << 53) as f64)
    }

    fn signed(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }

    fn complex(&mut self) -> Complex64 {
        Complex64::new(self.signed(), self.signed())
    }
}

fn hermitian_block(size: usize, values: &mut DeterministicValues) -> ComplexMatrix {
    let mut block = ComplexMatrix::zeros(size, size);
    for row in 0..size {
        block
            .set(row, row, Complex64::new(values.signed(), 0.0))
            .unwrap();
        for column in 0..row {
            let value = values.complex();
            block.set(row, column, value).unwrap();
            block.set(column, row, value.conj()).unwrap();
        }
    }
    block
}

fn rectangular_block(
    rows: usize,
    columns: usize,
    values: &mut DeterministicValues,
) -> ComplexMatrix {
    ComplexMatrix::new(
        rows,
        columns,
        (0..rows * columns).map(|_| values.complex()).collect(),
    )
    .unwrap()
}

fn generated_model(seed: u64) -> TightBindingModel {
    let mut values = DeterministicValues::new(seed);
    let lattice = Lattice::new(
        vec![
            vec![1.0 + 0.1 * values.unit(), 0.2 * values.signed()],
            vec![0.1 * values.signed(), 1.1 + 0.1 * values.unit()],
        ],
        vec![0, 1],
    )
    .unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let first = builder
        .add_orbital("first", [values.unit(), values.unit()])
        .unwrap();
    let spinor = builder
        .add_orbital_with_dof("spinor", [values.unit(), values.unit()], 2)
        .unwrap();
    let third = builder
        .add_orbital("third", [values.unit(), values.unit()])
        .unwrap();
    builder.set_onsite(first, values.signed()).unwrap();
    builder
        .set_onsite_block(spinor, hermitian_block(2, &mut values))
        .unwrap();
    builder.set_onsite(third, values.signed()).unwrap();
    builder
        .add_hopping_block(first, spinor, [0, 0], rectangular_block(1, 2, &mut values))
        .unwrap();
    builder
        .add_hopping_block(spinor, third, [1, 0], rectangular_block(2, 1, &mut values))
        .unwrap();
    builder
        .add_hopping_block(spinor, spinor, [0, 1], rectangular_block(2, 2, &mut values))
        .unwrap();
    builder
        .add_hopping(
            first,
            third,
            [-1, 1],
            Complex64::new(values.signed(), values.signed()),
        )
        .unwrap();
    builder.build().unwrap()
}

fn dense(matrix: &ComplexMatrix) -> DMatrix<Complex64> {
    DMatrix::from_row_slice(matrix.rows(), matrix.columns(), matrix.as_slice())
}

#[test]
fn generated_multiorbital_models_obey_spectral_and_derivative_invariants() {
    let momenta = [[0.0, 0.0], [0.137, 0.291], [0.49, 0.73]];
    for seed in 0..16 {
        let model = generated_model(seed);
        for momentum in momenta {
            let hamiltonian = model.hamiltonian(&momentum).unwrap();
            assert!(hamiltonian.is_hermitian(1.0e-12).unwrap());
            let eigensystem = model.eigensystem(&momentum).unwrap();
            assert!(eigensystem
                .eigenvalues()
                .windows(2)
                .all(|pair| pair[0] <= pair[1]));

            let matrix = dense(&hamiltonian);
            let vectors = dense(eigensystem.eigenvectors());
            let diagonal = DMatrix::from_diagonal(&DVector::from_iterator(
                eigensystem.eigenvalues().len(),
                eigensystem
                    .eigenvalues()
                    .iter()
                    .map(|value| Complex64::new(*value, 0.0)),
            ));
            let residual = &matrix * &vectors - &vectors * diagonal;
            assert!(residual.norm() < 1.0e-10 * (1.0 + matrix.norm()));
            let orthogonality = vectors.adjoint() * &vectors - DMatrix::<Complex64>::identity(4, 4);
            assert!(orthogonality.norm() < 1.0e-10);
            let trace = (0..4).map(|index| matrix[(index, index)].re).sum::<f64>();
            assert!((trace - eigensystem.eigenvalues().iter().sum::<f64>()).abs() < 1.0e-10);

            let derivatives = model.reduced_momentum_derivatives(&momentum).unwrap();
            let step = 2.0e-6;
            for axis in 0..2 {
                let mut plus = momentum;
                let mut minus = momentum;
                plus[axis] += step;
                minus[axis] -= step;
                let plus = model.hamiltonian(&plus).unwrap();
                let minus = model.hamiltonian(&minus).unwrap();
                let numerical = plus
                    .as_slice()
                    .iter()
                    .zip(minus.as_slice())
                    .map(|(plus, minus)| (plus - minus) / (2.0 * step))
                    .collect::<Vec<_>>();
                let error = numerical
                    .iter()
                    .zip(derivatives[axis].as_slice())
                    .map(|(actual, expected)| (*actual - expected).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                assert!(error < 1.0e-7 * (1.0 + dense(&derivatives[axis]).norm()));
            }

            let translated = model
                .eigensystem(&[momentum[0] + 1.0, momentum[1] - 2.0])
                .unwrap();
            for (reference, translated) in eigensystem
                .eigenvalues()
                .iter()
                .zip(translated.eigenvalues())
            {
                assert!((reference - translated).abs() < 1.0e-10);
            }
        }
    }
}
