use std::f64::consts::PI;

use thouless::matrix::ComplexMatrix;
use thouless::model::{Lattice, ModelBuilder};
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
