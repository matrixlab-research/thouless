use thouless::model::{Lattice, ModelBuilder};
use thouless::transform::{
    change_nonperiodic_vector, make_finite_cluster, make_finite_geometry, make_supercell,
    remove_orbitals, replace_onsite_blocks, FiniteSite, ModelTransformError,
};
use thouless::{Complex64, ComplexMatrix};

#[test]
fn removing_orbitals_compacts_endpoints_and_discards_incident_hoppings() {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let first = builder.add_orbital("first", [0.0]).unwrap();
    let removed = builder.add_orbital("removed", [0.25]).unwrap();
    let last = builder.add_orbital("last", [0.5]).unwrap();
    builder.set_onsite(first, -1.0).unwrap();
    builder.set_onsite(removed, 3.0).unwrap();
    builder.set_onsite(last, 2.0).unwrap();
    builder
        .add_hopping(first, removed, [0], Complex64::new(0.4, 0.0))
        .unwrap();
    builder
        .add_hopping(last, first, [1], Complex64::new(-0.7, 0.2))
        .unwrap();
    let model = builder.build().unwrap();

    let transformed = remove_orbitals(&model, &[1]).unwrap();
    assert_eq!(transformed.orbitals().len(), 2);
    assert_eq!(transformed.orbitals()[0].label(), "first");
    assert_eq!(transformed.orbitals()[1].label(), "last");
    assert_eq!(transformed.hoppings().len(), 1);
    assert_eq!(transformed.hoppings()[0].target().index(), 1);
    assert_eq!(transformed.hoppings()[0].source().index(), 0);
    assert_eq!(transformed.hoppings()[0].cell_offset(), &[1]);
    assert_eq!(transformed.state_count(), 2);
}

#[test]
fn removing_every_orbital_is_rejected() {
    let lattice = Lattice::finite(1).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    builder.add_orbital("only", [0.0]).unwrap();
    let model = builder.build().unwrap();
    assert!(matches!(
        remove_orbitals(&model, &[0]),
        Err(ModelTransformError::EmptyResult)
    ));
}

#[test]
fn changing_an_open_vector_preserves_cartesian_orbital_geometry_and_spectrum() {
    let height = 3.0_f64.sqrt() / 2.0;
    let lattice = Lattice::new(vec![vec![1.0, 0.0], vec![0.5, height]], vec![0]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let first = builder.add_orbital("a", [1.0 / 3.0, 1.0 / 3.0]).unwrap();
    let second = builder.add_orbital("b", [2.0 / 3.0, 2.0 / 3.0]).unwrap();
    builder.set_onsite(first, -0.4).unwrap();
    builder.set_onsite(second, 0.4).unwrap();
    builder
        .add_hopping(second, first, [1, 0], Complex64::new(-1.0, 0.0))
        .unwrap();
    let model = builder.build().unwrap();

    let transformed = change_nonperiodic_vector(&model, 1, None, false).unwrap();
    let new_open = &transformed.lattice().primitive_vectors()[1];
    assert!(new_open[0].abs() < 1.0e-12);
    assert!((new_open[1] - 1.0).abs() < 1.0e-12);
    for (old, new) in model.orbitals().iter().zip(transformed.orbitals()) {
        let old_cartesian = [
            old.reduced_position()[0] + 0.5 * old.reduced_position()[1],
            height * old.reduced_position()[1],
        ];
        let new_cartesian = [new.reduced_position()[0], new.reduced_position()[1]];
        assert!((old_cartesian[0] - new_cartesian[0]).abs() < 1.0e-12);
        assert!((old_cartesian[1] - new_cartesian[1]).abs() < 1.0e-12);
    }
    let old_spectrum = model.eigensystem(&[0.37]).unwrap();
    let new_spectrum = transformed.eigensystem(&[0.37]).unwrap();
    for (old, new) in old_spectrum
        .eigenvalues()
        .iter()
        .zip(new_spectrum.eigenvalues())
    {
        assert!((old - new).abs() < 1.0e-12);
    }
}

#[test]
fn supercell_spectrum_matches_folded_primitive_bands() {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder.add_orbital("site", [0.0]).unwrap();
    builder
        .add_hopping(orbital, orbital, [1], Complex64::new(-1.0, 0.0))
        .unwrap();
    let primitive = builder.build().unwrap();

    let result = make_supercell(&primitive, &[vec![2]], true).unwrap();
    assert_eq!(result.translations(), &[vec![0], vec![1]]);
    assert_eq!(result.model().orbitals().len(), 2);
    let supercell_spectrum = result.model().eigensystem(&[0.3]).unwrap();
    let first_fold = primitive.eigensystem(&[0.15]).unwrap();
    let second_fold = primitive.eigensystem(&[0.65]).unwrap();
    let mut expected = vec![first_fold.eigenvalues()[0], second_fold.eigenvalues()[0]];
    expected.sort_by(f64::total_cmp);
    for (actual, expected) in supercell_spectrum.eigenvalues().iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn finite_chain_has_open_boundaries_and_the_analytic_spectrum() {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder.add_orbital("site", [0.0]).unwrap();
    builder
        .add_hopping(orbital, orbital, [1], Complex64::new(-1.0, 0.0))
        .unwrap();
    let periodic = builder.build().unwrap();
    let cells = (0..5).map(|cell| vec![cell]).collect::<Vec<_>>();
    let finite = make_finite_cluster(&periodic, &cells).unwrap();

    assert_eq!(finite.model().lattice().periodic_dimension(), 0);
    assert_eq!(finite.model().orbitals().len(), 5);
    assert_eq!(finite.model().hoppings().len(), 4);
    assert_eq!(finite.sites()[3], FiniteSite::new([3], 0));
    let spectrum = finite.model().eigensystem(&[]).unwrap();
    for (mode, actual) in spectrum.eigenvalues().iter().enumerate() {
        let expected = -2.0 * (std::f64::consts::PI * (mode + 1) as f64 / 6.0).cos();
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn explicit_finite_sites_support_incomplete_cells_and_vacancies() {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let a = builder.add_orbital("a", [0.0]).unwrap();
    let b = builder.add_orbital("b", [0.5]).unwrap();
    builder
        .add_hopping(a, b, [0], Complex64::new(-1.0, 0.0))
        .unwrap();
    builder
        .add_hopping(a, b, [1], Complex64::new(-0.7, 0.0))
        .unwrap();
    let model = builder.build().unwrap();
    let sites = vec![
        FiniteSite::new([0], a.index()),
        FiniteSite::new([0], b.index()),
        FiniteSite::new([1], a.index()),
    ];
    let finite = make_finite_geometry(&model, &sites).unwrap();

    assert_eq!(finite.sites(), sites);
    assert_eq!(finite.model().orbitals().len(), 3);
    assert_eq!(finite.model().hoppings().len(), 1);
    assert_eq!(finite.model().orbitals()[2].reduced_position(), &[1.0]);

    let chain = {
        let mut builder = ModelBuilder::new(Lattice::new(vec![vec![1.0]], vec![0]).unwrap());
        let site = builder.add_orbital("site", [0.0]).unwrap();
        builder
            .add_hopping(site, site, [1], Complex64::new(-1.0, 0.0))
            .unwrap();
        builder.build().unwrap()
    };
    let cluster =
        make_finite_cluster(&chain, &(0..5).map(|cell| vec![cell]).collect::<Vec<_>>()).unwrap();
    let vacancy = remove_orbitals(cluster.model(), &[2]).unwrap();
    let spectrum = vacancy.eigensystem(&[]).unwrap();
    assert_eq!(spectrum.eigenvalues(), &[-1.0, -1.0, 1.0, 1.0]);
}

#[test]
fn onsite_replacements_add_disorder_without_changing_connectivity() {
    let lattice = Lattice::finite(1).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let first = builder.add_orbital("first", [0.0]).unwrap();
    let second = builder.add_orbital("second", [1.0]).unwrap();
    builder
        .add_hopping(first, second, [0], Complex64::new(-0.5, 0.2))
        .unwrap();
    let clean = builder.build().unwrap();
    let disordered = replace_onsite_blocks(
        &clean,
        &[
            (0, ComplexMatrix::scalar(Complex64::new(-0.7, 0.0))),
            (1, ComplexMatrix::scalar(Complex64::new(0.4, 0.0))),
        ],
    )
    .unwrap();

    assert_eq!(disordered.hoppings(), clean.hoppings());
    assert_eq!(
        disordered.onsite_blocks()[0].as_slice(),
        &[Complex64::new(-0.7, 0.0)]
    );
    assert_eq!(
        disordered.onsite_blocks()[1].as_slice(),
        &[Complex64::new(0.4, 0.0)]
    );
}
