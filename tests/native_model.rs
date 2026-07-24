use thouless::model::{Lattice, ModelBuilder};
use thouless::{Complex64, ModelError};

#[test]
fn one_model_represents_periodic_and_finite_translation_structures() {
    let periodic = Lattice::new(2, vec![vec![1.0, 0.0]]).expect("valid wire lattice");
    assert_eq!(periodic.real_dimension(), 2);
    assert_eq!(periodic.periodic_dimension(), 1);

    let finite = Lattice::new(2, Vec::new()).expect("valid finite geometry");
    assert_eq!(finite.periodic_dimension(), 0);
}

#[test]
fn builder_preserves_orbital_identity_and_hermitian_hopping_convention() {
    let lattice = Lattice::new(1, vec![vec![1.0]]).expect("valid chain lattice");
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder
        .add_orbital("s", [0.0])
        .expect("valid orbital");
    builder.set_onsite(orbital, 0.25).expect("valid onsite");
    builder
        .add_hopping(orbital, orbital, [1], Complex64::new(-1.0, 0.5))
        .expect("valid hopping");

    let model = builder.build().expect("non-empty model");
    assert_eq!(model.orbitals()[orbital.index()].label(), "s");
    assert_eq!(model.onsite(orbital), Some(0.25));
    assert_eq!(model.hoppings().len(), 1);
    assert_eq!(model.hoppings()[0].cell_offset(), &[1]);
}

#[test]
fn reverse_hopping_is_rejected_because_hermitian_partner_is_implicit() {
    let lattice = Lattice::new(1, vec![vec![1.0]]).expect("valid chain lattice");
    let mut builder = ModelBuilder::new(lattice);
    let left = builder
        .add_orbital("left", [0.0])
        .expect("valid orbital");
    let right = builder
        .add_orbital("right", [0.5])
        .expect("valid orbital");
    builder
        .add_hopping(left, right, [1], Complex64::new(-1.0, 0.25))
        .expect("first hopping is valid");

    let error = builder
        .add_hopping(right, left, [-1], Complex64::new(-1.0, -0.25))
        .expect_err("reverse term must not be stored twice");
    assert_eq!(error, ModelError::DuplicateHopping);
}

#[test]
fn malformed_dimensions_and_non_finite_values_are_rejected() {
    assert_eq!(
        Lattice::new(0, Vec::new()).expect_err("zero-dimensional embedding is invalid"),
        ModelError::InvalidRealDimension
    );
    assert!(matches!(
        Lattice::new(2, vec![vec![1.0]]),
        Err(ModelError::InvalidTranslationVector {
            expected: 2,
            actual: 1,
            ..
        })
    ));

    let lattice = Lattice::new(1, Vec::new()).expect("valid finite lattice");
    let mut builder = ModelBuilder::new(lattice);
    assert_eq!(
        builder
            .add_orbital("bad", [f64::NAN])
            .expect_err("NaN must be rejected"),
        ModelError::NonFiniteValue {
            field: "orbital position"
        }
    );
}
