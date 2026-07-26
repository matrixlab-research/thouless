use thouless::transport::{
    open_system_self_energies, partition_shot_noise, regularize_retarded_self_energy,
    retarded_lead_self_energy, solve_open_system, solve_open_system_from_self_energies,
    square_lattice_self_energy, surface_green_function, LeadContact, SurfaceGreenOptions,
};
use thouless::{Complex64, ComplexMatrix};

#[test]
fn scalar_chain_surface_green_matches_the_retarded_analytic_branch() {
    let onsite = ComplexMatrix::scalar(Complex64::new(0.0, 0.0));
    let hopping = ComplexMatrix::scalar(Complex64::new(-1.0, 0.0));
    let options = SurfaceGreenOptions::default();
    let surface = surface_green_function(&onsite, &hopping, 0.0, options).unwrap();
    let value = surface.get(0, 0).unwrap();
    assert!(value.re.abs() < 1.0e-8);
    assert!(
        (value.im + 1.0).abs() < 5.0e-5,
        "surface Green value was {value}"
    );
}

#[test]
fn scalar_chain_self_energy_reaches_the_zero_broadening_limit() {
    let onsite = ComplexMatrix::scalar(Complex64::new(0.3, 0.0));
    let hopping = ComplexMatrix::scalar(Complex64::new(0.7, 0.0));
    let self_energy =
        retarded_lead_self_energy(&onsite, &hopping, 0.0, Default::default()).unwrap();
    let expected = Complex64::new(-0.15, -(1.87_f64).sqrt() / 2.0);
    assert!(
        (self_energy.get(0, 0).unwrap() - expected).norm() < 1.0e-9,
        "self-energy was {} instead of {expected}",
        self_energy.get(0, 0).unwrap(),
    );
}

#[test]
fn rectangular_interface_returns_an_interface_sized_self_energy() {
    let onsite = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.3, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        ],
    )
    .unwrap();
    let hopping = ComplexMatrix::new(
        2,
        1,
        vec![Complex64::new(0.7, 0.0), Complex64::new(0.0, 0.0)],
    )
    .unwrap();
    let self_energy =
        retarded_lead_self_energy(&onsite, &hopping, 0.0, Default::default()).unwrap();
    assert_eq!(self_energy.shape(), (1, 1));
    let expected = Complex64::new(-0.15, -(1.87_f64).sqrt() / 2.0);
    assert!((self_energy.get(0, 0).unwrap() - expected).norm() < 1.0e-9);
}

#[test]
fn retarded_regularization_clamps_negative_broadening_channels() {
    let self_energy = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(1.0, -2.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.5),
        ],
    )
    .unwrap();
    let regularized = regularize_retarded_self_energy(&self_energy, None).unwrap();
    assert!((regularized.get(0, 0).unwrap() - Complex64::new(1.0, -2.0)).norm() < 1.0e-12);
    assert!((regularized.get(1, 1).unwrap() - Complex64::new(-1.0, 0.0)).norm() < 1.0e-12);
}

#[test]
fn square_strip_closed_form_matches_general_surface_decimation() {
    let width = 5;
    let hopping = 0.78;
    let fermi_energy = 1.3;
    let mut cell = ComplexMatrix::zeros(width, width);
    for site in 0..width {
        cell.set(
            site,
            site,
            Complex64::new(4.0 * hopping - fermi_energy, 0.0),
        )
        .unwrap();
        if site + 1 < width {
            cell.set(site, site + 1, Complex64::new(-hopping, 0.0))
                .unwrap();
            cell.set(site + 1, site, Complex64::new(-hopping, 0.0))
                .unwrap();
        }
    }
    let mut inter_cell = ComplexMatrix::zeros(width, width);
    for site in 0..width {
        inter_cell
            .set(site, site, Complex64::new(-hopping, 0.0))
            .unwrap();
    }
    let analytic = square_lattice_self_energy(width, hopping, fermi_energy).unwrap();
    let numerical = retarded_lead_self_energy(&cell, &inter_cell, 0.0, Default::default()).unwrap();
    for (analytic, numerical) in analytic.as_slice().iter().zip(numerical.as_slice()) {
        assert!((analytic - numerical).norm() < 1.0e-9);
    }
}

#[test]
fn matched_one_site_device_has_unit_ballistic_transmission() {
    let onsite = ComplexMatrix::scalar(Complex64::new(0.0, 0.0));
    let hopping = ComplexMatrix::scalar(Complex64::new(-1.0, 0.0));
    let coupling = ComplexMatrix::scalar(Complex64::new(-1.0, 0.0));
    let contact = LeadContact::new(onsite.clone(), hopping, coupling).unwrap();
    let contacts = [contact.clone(), contact];
    let self_energies =
        open_system_self_energies(&onsite, &contacts, 0.0, Default::default()).unwrap();
    let solution = solve_open_system(&onsite, &contacts, 0.0, Default::default()).unwrap();
    assert_eq!(self_energies, solution.self_energies());
    assert!((solution.transmission(1, 0).unwrap() - 1.0).abs() < 2.0e-6);
    assert!((solution.transmission(0, 1).unwrap() - 1.0).abs() < 2.0e-6);
}

#[test]
fn partition_noise_is_sum_of_channel_binomial_variances() {
    let reflection = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.5_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.8_f64.sqrt(), 0.0),
        ],
    )
    .unwrap();
    let expected = 0.5 * 0.5 + 0.8 * 0.2;
    assert!((partition_shot_noise(&reflection).unwrap() - expected).abs() < 1.0e-12);
}

#[test]
fn embedded_self_energy_solution_obeys_spectral_and_fisher_lee_identities() {
    let device = ComplexMatrix::new(
        3,
        3,
        vec![
            0.2.into(),
            (-0.8).into(),
            0.0.into(),
            (-0.8).into(),
            (-0.1).into(),
            (-0.6).into(),
            0.0.into(),
            (-0.6).into(),
            0.3.into(),
        ],
    )
    .unwrap();
    let gamma = 1.4;
    let mut left = ComplexMatrix::zeros(3, 3);
    left.set(0, 0, Complex64::new(0.0, -gamma / 2.0)).unwrap();
    let mut right = ComplexMatrix::zeros(3, 3);
    right.set(2, 2, Complex64::new(0.0, -gamma / 2.0)).unwrap();
    let solution = solve_open_system_from_self_energies(&device, &[left, right], 0.15).unwrap();

    let green = solution.retarded_green();
    let total_gamma =
        solution
            .broadenings()
            .iter()
            .fold(ComplexMatrix::zeros(3, 3), |mut total, matrix| {
                for row in 0..3 {
                    for column in 0..3 {
                        total
                            .add_entry(row, column, matrix.get(row, column).unwrap())
                            .unwrap();
                    }
                }
                total
            });
    for row in 0..3 {
        for column in 0..3 {
            let spectral = green.get(row, column).unwrap() - green.get(column, row).unwrap().conj();
            let mut dissipative = Complex64::new(0.0, 0.0);
            for first in 0..3 {
                for second in 0..3 {
                    dissipative += green.get(row, first).unwrap()
                        * total_gamma.get(first, second).unwrap()
                        * green.get(column, second).unwrap().conj();
                }
            }
            assert!((spectral + Complex64::i() * dissipative).norm() < 1.0e-11);
        }
    }

    let factor = |orbital| {
        let mut value = ComplexMatrix::zeros(3, 1);
        value
            .set(orbital, 0, Complex64::new(gamma.sqrt(), 0.0))
            .unwrap();
        value
    };
    let factors = [factor(0), factor(2)];
    let scattering = solution
        .scattering_matrix(&factors, &factors, 1.0e-13)
        .unwrap();
    let matrix = scattering.matrix();
    for first in 0..2 {
        for second in 0..2 {
            let overlap = (0..2)
                .map(|row| {
                    matrix.get(row, first).unwrap().conj() * matrix.get(row, second).unwrap()
                })
                .sum::<Complex64>();
            let expected = if first == second {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert!((overlap - expected).norm() < 1.0e-10);
        }
    }

    let counts = solution.channel_counts(&[Some(1), None], 1.0e-10).unwrap();
    assert_eq!(counts, [1, 1]);
    let green_values = solution
        .green_function_transmission_matrix(&counts)
        .unwrap();
    let scattering_values = scattering.transmission_matrix();
    for drain in 0..2 {
        for source in 0..2 {
            let block = scattering.block(drain, source).unwrap();
            let probability = block
                .as_slice()
                .iter()
                .map(Complex64::norm_sqr)
                .sum::<f64>();
            assert!((scattering_values[drain][source] - probability).abs() < 1.0e-14);
            assert!((green_values[drain][source] - probability).abs() < 1.0e-10);
        }
    }

    let states = solution.scattering_states(&factors).unwrap();
    assert_eq!(states[0].shape(), (1, 3));
    for orbital in 0..3 {
        let expected = green.get(orbital, 0).unwrap() * gamma.sqrt();
        assert!((states[0].get(0, orbital).unwrap() - expected).norm() < 1.0e-12);
    }
    assert!(solution
        .local_density_of_states()
        .iter()
        .all(|value| *value >= 0.0));
}

#[test]
fn generated_multiterminal_devices_conserve_scattering_probability() {
    for family in 0..8 {
        let dimension = 3 + family % 4;
        let mut device = ComplexMatrix::zeros(dimension, dimension);
        for row in 0..dimension {
            device
                .set(row, row, Complex64::new(0.07 * (row + family) as f64, 0.0))
                .unwrap();
            if row + 1 < dimension {
                let hopping = -0.4 - 0.03 * ((row + 2 * family) % 5) as f64;
                device.set(row, row + 1, hopping.into()).unwrap();
                device.set(row + 1, row, hopping.into()).unwrap();
            }
        }
        let contacts = [0, dimension / 2, dimension - 1];
        let mut self_energies = Vec::new();
        let mut factors = Vec::new();
        for (lead, orbital) in contacts.into_iter().enumerate() {
            let broadening = 0.8 + 0.1 * (lead + family) as f64;
            let mut self_energy = ComplexMatrix::zeros(dimension, dimension);
            self_energy
                .set(orbital, orbital, Complex64::new(0.0, -broadening / 2.0))
                .unwrap();
            self_energies.push(self_energy);
            let mut factor = ComplexMatrix::zeros(dimension, 1);
            factor
                .set(orbital, 0, Complex64::new(broadening.sqrt(), 0.0))
                .unwrap();
            factors.push(factor);
        }
        let solution = solve_open_system_from_self_energies(
            &device,
            &self_energies,
            -0.3 + 0.11 * family as f64,
        )
        .unwrap();
        let scattering = solution
            .scattering_matrix(&factors, &factors, 1.0e-13)
            .unwrap();
        for source in 0..3 {
            let total = (0..3)
                .map(|drain| {
                    scattering
                        .block(drain, source)
                        .unwrap()
                        .as_slice()
                        .iter()
                        .map(Complex64::norm_sqr)
                        .sum::<f64>()
                })
                .sum::<f64>();
            assert!((total - 1.0).abs() < 1.0e-10);
        }
    }
}
