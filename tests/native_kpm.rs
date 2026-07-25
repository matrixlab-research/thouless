use thouless::kpm::{
    chebyshev_vectors, correlation_integral_factor, correlation_moments, correlation_response,
    integrate, reconstruct, rescale_hamiltonian, scalar_moments, Kernel,
};
use thouless::{Complex64, ComplexMatrix};

fn three_site_chain() -> ComplexMatrix {
    ComplexMatrix::new(
        3,
        3,
        vec![
            Complex64::new(0.2, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(-0.1, 0.0),
            Complex64::new(-0.7, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-0.7, 0.0),
            Complex64::new(0.4, 0.0),
        ],
    )
    .unwrap()
}

fn local_basis(dimension: usize) -> Vec<Vec<Complex64>> {
    (0..dimension)
        .map(|index| {
            let mut vector = vec![Complex64::new(0.0, 0.0); dimension];
            vector[index] = Complex64::new(1.0, 0.0);
            vector
        })
        .collect()
}

#[test]
fn local_spectral_channels_obey_the_trace_sum_rule() {
    let rescaled = rescale_hamiltonian(&three_site_chain(), 0.05, None).unwrap();
    let basis = local_basis(3);
    let vectors = chebyshev_vectors(rescaled.matrix(), &basis, 192).unwrap();
    let moments = scalar_moments(&basis, &vectors, None).unwrap();
    let spectrum = reconstruct(&moments, rescaled.scale(), Kernel::Jackson, false).unwrap();
    let integrated = integrate(
        spectrum.gammas(),
        &vec![1.0; spectrum.energies().len()],
        rescaled.scale(),
    )
    .unwrap();
    let trace = integrated.iter().map(|channel| channel[0].re).sum::<f64>();
    assert!((trace - 3.0).abs() < 1.0e-10);
    assert!(integrated
        .iter()
        .all(|channel| (channel[0].re - 1.0).abs() < 1.0e-10));
}

#[test]
fn explicit_and_computed_bounds_define_the_same_rescaling() {
    let hamiltonian = three_site_chain();
    let computed = rescale_hamiltonian(&hamiltonian, 0.05, None).unwrap();
    let scale = computed.scale();
    let strict_width = scale.half_width() * (2.0 - 0.05);
    let bounds = (
        scale.center() - strict_width / 2.0,
        scale.center() + strict_width / 2.0,
    );
    let explicit = rescale_hamiltonian(&hamiltonian, 0.05, Some(bounds)).unwrap();
    for (left, right) in computed
        .matrix()
        .as_slice()
        .iter()
        .zip(explicit.matrix().as_slice())
    {
        assert!((*left - *right).norm() < 1.0e-12);
    }
}

#[test]
fn a_zero_perturbation_has_zero_kubo_response() {
    let rescaled = rescale_hamiltonian(&three_site_chain(), 0.05, None).unwrap();
    let basis = local_basis(3);
    let left = chebyshev_vectors(rescaled.matrix(), &basis, 24).unwrap();
    let right = vec![vec![vec![Complex64::new(0.0, 0.0); 3]; 24]; 3];
    let moments = correlation_moments(&left, &right, true).unwrap();
    let factor = correlation_integral_factor(&moments, 24, Kernel::Jackson).unwrap();
    for temperature in [0.0, 0.03, 0.2] {
        let response = correlation_response(&factor, rescaled.scale(), 0.1, temperature).unwrap();
        assert_eq!(response, vec![Complex64::new(0.0, 0.0)]);
    }
}
