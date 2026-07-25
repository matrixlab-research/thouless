use thouless::transport::{
    partition_shot_noise, regularize_retarded_self_energy, retarded_lead_self_energy,
    solve_open_system, square_lattice_self_energy, surface_green_function, LeadContact,
    SurfaceGreenOptions,
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
    let solution = solve_open_system(
        &onsite,
        &[contact.clone(), contact],
        0.0,
        Default::default(),
    )
    .unwrap();
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
