use std::f64::consts::TAU;

use thouless::ad::{AffineHermitianFamily, ModelParameters, SpectralProjectorObjective};
use thouless::model::{Lattice, ModelBuilder, TightBindingModel};
use thouless::observables::project_diagonal_observable;
use thouless::topology::{
    chern_numbers_on_uniform_grid, reduced_polarization_on_loop, wilson_line_phase,
};
use thouless::transform::{make_finite_geometry, FiniteSite};
use thouless::transport::{solve_open_system, LeadContact};
use thouless::{Complex64, ComplexMatrix};

fn matrix(values: &[(f64, f64)]) -> ComplexMatrix {
    let dimension = (values.len() as f64).sqrt() as usize;
    ComplexMatrix::new(
        dimension,
        dimension,
        values
            .iter()
            .map(|(real, imaginary)| Complex64::new(*real, *imaginary))
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
            .map(|(left, right)| *left * left_factor + *right * right_factor)
            .collect(),
    )
    .unwrap()
}

fn ssh_model() -> TightBindingModel {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let first = builder.add_orbital("a", [0.0]).unwrap();
    let second = builder.add_orbital("b", [0.5]).unwrap();
    builder
        .add_hopping(first, second, [0], Complex64::new(0.6, 0.0))
        .unwrap();
    builder
        .add_hopping(first, second, [1], Complex64::new(1.0, 0.0))
        .unwrap();
    builder.build().unwrap()
}

fn qwz_model() -> TightBindingModel {
    let lattice = Lattice::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0, 1]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder
        .add_orbital_with_dof("spinor", [0.0, 0.0], 2)
        .unwrap();
    let sigma_x = matrix(&[(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]);
    let sigma_y = matrix(&[(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)]);
    let sigma_z = matrix(&[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)]);
    builder
        .set_onsite_block(
            orbital,
            add_scaled(&sigma_z, -1.0, &sigma_z, Complex64::new(0.0, 0.0)),
        )
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
    builder.build().unwrap()
}

fn frame(values: &[Complex64]) -> ComplexMatrix {
    ComplexMatrix::new(1, values.len(), values.to_vec()).unwrap()
}

fn main() {
    let ad_family = AffineHermitianFamily::new(
        matrix(&[(-1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]),
        vec![matrix(&[(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)])],
    )
    .unwrap();
    let ad_objective = SpectralProjectorObjective::new(
        &ad_family,
        1,
        matrix(&[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]),
        1.0e-8,
    )
    .unwrap();
    let (ad_value, ad_gradient) = ad_objective
        .value_and_grad(&ModelParameters::new(vec![0.2]).unwrap())
        .unwrap();

    let ssh = ssh_model();
    let zone_edge = ssh.eigensystem(&[0.5]).unwrap();
    let ssh_gap = zone_edge.eigenvalues()[1] - zone_edge.eigenvalues()[0];
    let ssh_polarization = reduced_polarization_on_loop(&ssh, 65, 0, &[0.0], &[0]).unwrap();

    let qwz = qwz_model();
    let chern = chern_numbers_on_uniform_grid(&qwz, &[31, 31], [0, 1], &[0]).unwrap();

    let finite = make_finite_geometry(&ssh, &[FiniteSite::new([0], 0), FiniteSite::new([2], 1)])
        .unwrap()
        .into_model();
    let vacancy_states = finite.state_count();
    let projected =
        project_diagonal_observable(&ComplexMatrix::identity(vacancy_states), &[1.0, 2.0]).unwrap();
    let vacancy_trace = (0..vacancy_states)
        .map(|index| projected.get(index, index).unwrap().re)
        .sum::<f64>();

    let onsite = ComplexMatrix::scalar(Complex64::new(0.0, 0.0));
    let hopping = ComplexMatrix::scalar(Complex64::new(-1.0, 0.0));
    let lead = LeadContact::new(onsite.clone(), hopping.clone(), hopping).unwrap();
    let scattering =
        solve_open_system(&onsite, &[lead.clone(), lead], 0.0, Default::default()).unwrap();

    let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
    let frames = vec![
        frame(&[
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(inverse_sqrt_two, 0.0),
        ]),
        frame(&[
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(0.0, inverse_sqrt_two),
        ]),
        frame(&[
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(inverse_sqrt_two, 0.0),
        ]),
    ];
    let phase = wilson_line_phase(&frames).unwrap();
    let gauge = Complex64::from_polar(1.0, 0.37);
    let transformed = vec![
        frames[0].clone(),
        frame(
            &frames[1]
                .as_slice()
                .iter()
                .map(|value| gauge * value)
                .collect::<Vec<_>>(),
        ),
        frames[2].clone(),
    ];
    let gauge_delta = (wilson_line_phase(&transformed).unwrap() - phase).abs();
    let invalid_shape = if ssh.hamiltonian(&[0.0, 0.5]).is_err() {
        1.0
    } else {
        0.0
    };

    let metrics = [
        ("ad_projector_value", ad_value),
        ("ad_projector_gradient", ad_gradient.as_slice()[0]),
        ("ssh_gap", ssh_gap),
        (
            "ssh_polarization",
            (ssh_polarization * TAU).rem_euclid(TAU) / TAU,
        ),
        ("chern_absolute", chern.values()[0].abs()),
        ("vacancy_states", vacancy_states as f64),
        ("vacancy_observable_trace", vacancy_trace),
        (
            "ballistic_transmission",
            scattering.transmission(1, 0).unwrap(),
        ),
        ("wilson_gauge_delta", gauge_delta),
        ("invalid_shape_error", invalid_shape),
    ];
    for (name, value) in metrics {
        println!("{name}={value:.17e}");
    }
}
