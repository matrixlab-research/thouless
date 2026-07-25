use thouless::transport::{
    partition_shot_noise, solve_open_system, surface_green_function, LeadContact,
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
