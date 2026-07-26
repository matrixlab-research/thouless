use nalgebra::DMatrix;
use thouless::model::{Lattice, ModelBuilder};
use thouless::response::{
    band_response_from_hamiltonian_derivatives, band_response_from_model, berry_curvature_dipole,
    intrinsic_berry_curvature_from_hamiltonian_derivatives, occupation_weighted_berry_curvature,
    uniform_mesh_intrinsic_berry_curvature, FermiDistribution, IntrinsicResponseError,
    MomentumCoordinates, UniformMeshBandResponse,
};
use thouless::{Complex64, ComplexMatrix};

fn massive_dirac(mass: f64, tilt_y: f64) -> (ComplexMatrix, ComplexMatrix, ComplexMatrix) {
    let hamiltonian = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(mass, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-mass, 0.0),
        ],
    )
    .unwrap();
    let sigma_x = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let tilted_sigma_y = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(tilt_y, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(tilt_y, 0.0),
        ],
    )
    .unwrap();
    (hamiltonian, sigma_x, tilted_sigma_y)
}

fn transform(matrix: &ComplexMatrix, unitary: &DMatrix<Complex64>) -> ComplexMatrix {
    let matrix = DMatrix::from_row_slice(matrix.rows(), matrix.columns(), matrix.as_slice());
    let transformed = unitary.adjoint() * matrix * unitary;
    ComplexMatrix::new(
        transformed.nrows(),
        transformed.ncols(),
        (0..transformed.nrows())
            .flat_map(|row| {
                let transformed = &transformed;
                (0..transformed.ncols()).map(move |column| transformed[(row, column)])
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn massive_dirac_response_matches_the_analytic_kubo_curvature() {
    let mass = 2.0;
    let (hamiltonian, sigma_x, sigma_y) = massive_dirac(mass, 0.0);
    let fermi = FermiDistribution::new(0.0, 0.0).unwrap();
    let point = band_response_from_hamiltonian_derivatives(
        &hamiltonian,
        &[sigma_x.clone(), sigma_y.clone()],
        fermi,
        1.0e-12,
    )
    .unwrap();

    let lower_curvature = -1.0 / (2.0 * mass * mass);
    assert_eq!(point.energies(), &[-mass, mass]);
    assert_eq!(point.occupations(), &[1.0, 0.0]);
    assert_eq!(point.negative_occupation_derivatives(), None);
    assert!((point.berry_curvature(0, 0, 1).unwrap() - lower_curvature).abs() < 1.0e-12);
    assert!((point.berry_curvature(1, 0, 1).unwrap() + lower_curvature).abs() < 1.0e-12);
    assert!((point.berry_curvature(0, 1, 0).unwrap() + lower_curvature).abs() < 1.0e-12);
    assert_eq!(point.berry_curvature(0, 0, 0), Some(0.0));

    let integral = occupation_weighted_berry_curvature(&[point], &[0.25], 0, 1).unwrap();
    assert!((integral - 0.25 * lower_curvature).abs() < 1.0e-12);
    let subspace_curvature = intrinsic_berry_curvature_from_hamiltonian_derivatives(
        &hamiltonian,
        &[sigma_x, sigma_y],
        fermi,
        1.0e-12,
    )
    .unwrap();
    assert!((subspace_curvature.get(0, 1).unwrap() - lower_curvature).abs() < 1.0e-12);
}

#[test]
fn response_is_invariant_under_a_constant_basis_change() {
    let (hamiltonian, sigma_x, sigma_y) = massive_dirac(1.7, 0.0);
    let fermi = FermiDistribution::new(0.2, 0.3).unwrap();
    let reference = band_response_from_hamiltonian_derivatives(
        &hamiltonian,
        &[sigma_x.clone(), sigma_y.clone()],
        fermi,
        1.0e-12,
    )
    .unwrap();

    let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
    let unitary = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(0.0, inverse_sqrt_two),
            Complex64::new(0.0, inverse_sqrt_two),
            Complex64::new(inverse_sqrt_two, 0.0),
        ],
    );
    let transformed = band_response_from_hamiltonian_derivatives(
        &transform(&hamiltonian, &unitary),
        &[transform(&sigma_x, &unitary), transform(&sigma_y, &unitary)],
        fermi,
        1.0e-12,
    )
    .unwrap();

    for band in 0..2 {
        assert!((reference.energies()[band] - transformed.energies()[band]).abs() < 1.0e-12);
        for direction in 0..2 {
            assert!(
                (reference.group_velocity(band, direction).unwrap()
                    - transformed.group_velocity(band, direction).unwrap())
                .abs()
                    < 1.0e-12
            );
            for second in 0..2 {
                assert!(
                    (reference.berry_curvature(band, direction, second).unwrap()
                        - transformed
                            .berry_curvature(band, direction, second)
                            .unwrap())
                    .abs()
                        < 1.0e-12
                );
            }
        }
    }
}

#[test]
fn degenerate_subspace_curvature_is_gauge_invariant() {
    let mass = 1.7_f64;
    let hamiltonian = ComplexMatrix::new(
        4,
        4,
        vec![
            Complex64::new(-mass, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-mass, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(mass, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(mass, 0.0),
        ],
    )
    .unwrap();
    let mut first_derivative = ComplexMatrix::zeros(4, 4);
    let mut second_derivative = ComplexMatrix::zeros(4, 4);
    for (occupied, empty) in [(0, 2), (1, 3)] {
        first_derivative
            .set(occupied, empty, Complex64::new(1.0, 0.0))
            .unwrap();
        first_derivative
            .set(empty, occupied, Complex64::new(1.0, 0.0))
            .unwrap();
        second_derivative
            .set(occupied, empty, Complex64::new(0.0, 1.0))
            .unwrap();
        second_derivative
            .set(empty, occupied, Complex64::new(0.0, -1.0))
            .unwrap();
    }
    let fermi = FermiDistribution::new(0.0, 0.0).unwrap();
    assert!(matches!(
        band_response_from_hamiltonian_derivatives(
            &hamiltonian,
            &[first_derivative.clone(), second_derivative.clone()],
            fermi,
            1.0e-12,
        ),
        Err(IntrinsicResponseError::DegenerateBands {
            first: 0,
            second: 1
        })
    ));

    let reference = intrinsic_berry_curvature_from_hamiltonian_derivatives(
        &hamiltonian,
        &[first_derivative.clone(), second_derivative.clone()],
        fermi,
        1.0e-12,
    )
    .unwrap();
    assert!((reference.get(0, 1).unwrap() + 1.0 / mass.powi(2)).abs() < 1.0e-12);
    assert!((reference.get(1, 0).unwrap() - 1.0 / mass.powi(2)).abs() < 1.0e-12);

    let occupied_angle = 0.37_f64;
    let empty_angle = -0.23_f64;
    let occupied_cosine = occupied_angle.cos();
    let occupied_sine = occupied_angle.sin();
    let empty_cosine = empty_angle.cos();
    let empty_sine = empty_angle.sin();
    let unitary = DMatrix::from_row_slice(
        4,
        4,
        &[
            Complex64::new(occupied_cosine, 0.0),
            Complex64::new(0.0, occupied_sine),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, occupied_sine),
            Complex64::new(occupied_cosine, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(empty_cosine, 0.0),
            Complex64::new(empty_sine, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-empty_sine, 0.0),
            Complex64::new(empty_cosine, 0.0),
        ],
    );
    let transformed = intrinsic_berry_curvature_from_hamiltonian_derivatives(
        &transform(&hamiltonian, &unitary),
        &[
            transform(&first_derivative, &unitary),
            transform(&second_derivative, &unitary),
        ],
        fermi,
        1.0e-12,
    )
    .unwrap();
    for (actual, expected) in transformed.as_slice().iter().zip(reference.as_slice()) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn tilted_dirac_point_produces_the_expected_fermi_surface_dipole() {
    let mass = 1.3;
    let tilt_y = 0.4;
    let temperature = 0.25;
    let (hamiltonian, sigma_x, tilted_sigma_y) = massive_dirac(mass, tilt_y);
    let fermi = FermiDistribution::new(-mass, temperature).unwrap();
    let point = band_response_from_hamiltonian_derivatives(
        &hamiltonian,
        &[sigma_x, tilted_sigma_y],
        fermi,
        1.0e-12,
    )
    .unwrap();

    let negative_derivative = 1.0 / (4.0 * temperature);
    let lower_curvature = -1.0 / (2.0 * mass * mass);
    let expected_lower = negative_derivative * tilt_y * lower_curvature;
    let upper = point.negative_occupation_derivatives().unwrap()[1] * tilt_y * -lower_curvature;
    let result = berry_curvature_dipole(&[point], &[1.0], 1, 0, 1).unwrap();
    assert!((result - (expected_lower + upper)).abs() < 1.0e-12);
}

#[test]
fn singular_band_and_zero_temperature_fermi_surface_are_explicit() {
    let zero = ComplexMatrix::zeros(2, 2);
    let identity = ComplexMatrix::identity(2);
    let fermi = FermiDistribution::new(0.0, 0.0).unwrap();
    assert_eq!(
        band_response_from_hamiltonian_derivatives(
            &zero,
            std::slice::from_ref(&identity),
            fermi,
            1.0e-12
        ),
        Err(IntrinsicResponseError::DegenerateBands {
            first: 0,
            second: 1
        })
    );

    let hamiltonian = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap();
    let point =
        band_response_from_hamiltonian_derivatives(&hamiltonian, &[identity], fermi, 1.0e-12)
            .unwrap();
    assert_eq!(
        berry_curvature_dipole(&[point], &[1.0], 0, 0, 0),
        Err(IntrinsicResponseError::ZeroTemperatureFermiSurface)
    );
}

#[test]
fn model_response_preserves_reduced_and_cartesian_derivative_coordinates() {
    let lattice = Lattice::new(vec![vec![2.0]], vec![0]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder.add_orbital("s", [0.0]).unwrap();
    builder.set_onsite(orbital, 0.2).unwrap();
    builder
        .add_hopping(orbital, orbital, [1], Complex64::new(-1.0, 0.0))
        .unwrap();
    let model = builder.build().unwrap();
    let momentum = [0.125];
    let fermi = FermiDistribution::new(0.0, 0.1).unwrap();

    let reduced = band_response_from_model(
        &model,
        &momentum,
        fermi,
        MomentumCoordinates::Reduced,
        1.0e-12,
    )
    .unwrap();
    let cartesian = band_response_from_model(
        &model,
        &momentum,
        fermi,
        MomentumCoordinates::Cartesian,
        1.0e-12,
    )
    .unwrap();

    let phase = std::f64::consts::TAU * momentum[0];
    assert!((reduced.energies()[0] - (0.2 - 2.0 * phase.cos())).abs() < 1.0e-12);
    assert!(
        (reduced.group_velocity(0, 0).unwrap() - 2.0 * std::f64::consts::TAU * phase.sin()).abs()
            < 1.0e-12
    );
    assert!((cartesian.group_velocity(0, 0).unwrap() - 4.0 * phase.sin()).abs() < 1.0e-12);
    assert_eq!(reduced.berry_curvature(0, 0, 0), Some(0.0));
}

fn scaled(matrix: &ComplexMatrix, factor: f64) -> ComplexMatrix {
    ComplexMatrix::new(
        matrix.rows(),
        matrix.columns(),
        matrix
            .as_slice()
            .iter()
            .map(|value| factor * value)
            .collect(),
    )
    .unwrap()
}

fn add_scaled(
    left: &ComplexMatrix,
    left_factor: f64,
    right: &ComplexMatrix,
    right_factor: Complex64,
) -> ComplexMatrix {
    ComplexMatrix::new(
        left.rows(),
        left.columns(),
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(left, right)| left_factor * left + right_factor * right)
            .collect(),
    )
    .unwrap()
}

fn qi_wu_zhang_model() -> thouless::model::TightBindingModel {
    qi_wu_zhang_model_with_copies(1)
}

fn qi_wu_zhang_model_with_copies(copies: usize) -> thouless::model::TightBindingModel {
    let lattice = Lattice::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0, 1]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let orbitals = (0..copies)
        .map(|copy| {
            builder
                .add_orbital_with_dof(format!("spinor-{copy}"), [0.0, 0.0], 2)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let sigma_x = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_y = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_z = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap();
    for orbital in orbitals {
        builder
            .set_onsite_block(orbital, scaled(&sigma_z, -1.0))
            .unwrap();
        builder
            .add_hopping_block(
                orbital,
                orbital,
                [1, 0],
                add_scaled(&sigma_z, 0.5, &sigma_x, Complex64::new(0.0, -0.5)),
            )
            .unwrap();
        builder
            .add_hopping_block(
                orbital,
                orbital,
                [0, 1],
                add_scaled(&sigma_z, 0.5, &sigma_y, Complex64::new(0.0, -0.5)),
            )
            .unwrap();
    }
    builder.build().unwrap()
}

#[test]
fn uniform_mesh_response_recovers_chern_integral_and_inversion_cancellation() {
    let model = qi_wu_zhang_model();
    let zero_temperature = FermiDistribution::new(0.0, 0.0).unwrap();
    let cartesian = UniformMeshBandResponse::from_model(
        &model,
        &[31, 31],
        &[0.0, 0.0],
        zero_temperature,
        MomentumCoordinates::Cartesian,
        1.0e-10,
    )
    .unwrap();
    let reduced = UniformMeshBandResponse::from_model(
        &model,
        &[31, 31],
        &[0.0, 0.0],
        zero_temperature,
        MomentumCoordinates::Reduced,
        1.0e-10,
    )
    .unwrap();
    let cartesian_integral = cartesian.occupation_weighted_berry_curvature(0, 1).unwrap();
    let reduced_integral = reduced.occupation_weighted_berry_curvature(0, 1).unwrap();
    assert!((cartesian_integral.abs() / std::f64::consts::TAU - 1.0).abs() < 1.0e-8);
    assert!((reduced_integral - cartesian_integral).abs() < 1.0e-10);

    let finite_temperature = FermiDistribution::new(-1.0, 0.2).unwrap();
    let centered = UniformMeshBandResponse::from_model(
        &model,
        &[24, 24],
        &[0.5, 0.5],
        finite_temperature,
        MomentumCoordinates::Cartesian,
        1.0e-10,
    )
    .unwrap();
    assert!(centered.berry_curvature_dipole(0, 0, 1).unwrap().abs() < 1.0e-12);
}

#[test]
fn uniform_intrinsic_curvature_handles_exact_spin_degeneracy() {
    let model = qi_wu_zhang_model_with_copies(2);
    let fermi = FermiDistribution::new(0.0, 0.0).unwrap();
    assert!(matches!(
        UniformMeshBandResponse::from_model(
            &model,
            &[7, 7],
            &[0.0, 0.0],
            fermi,
            MomentumCoordinates::Cartesian,
            1.0e-10,
        ),
        Err(IntrinsicResponseError::DegenerateBands { .. })
    ));

    let integrated = uniform_mesh_intrinsic_berry_curvature(
        &model,
        &[31, 31],
        &[0.0, 0.0],
        fermi,
        MomentumCoordinates::Cartesian,
        1.0e-10,
    )
    .unwrap();
    assert!((integrated.get(0, 1).unwrap().abs() / std::f64::consts::TAU - 2.0).abs() < 2.0e-8);
    assert!((integrated.get(0, 1).unwrap() + integrated.get(1, 0).unwrap()).abs() < 1.0e-10);
}
