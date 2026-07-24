use thouless::topology::{
    connection_from_link, parallel_transport_link, plaquette_flux, wilson_line_phase,
    wilson_loop_eigenphases,
};
use thouless::{Complex64, ComplexMatrix};

fn frame(values: &[Complex64]) -> ComplexMatrix {
    ComplexMatrix::new(1, values.len(), values.to_vec()).unwrap()
}

#[test]
fn wilson_phase_is_invariant_under_local_frame_phases() {
    let inv_sqrt_two = 1.0 / 2.0_f64.sqrt();
    let frames = vec![
        frame(&[
            Complex64::new(inv_sqrt_two, 0.0),
            Complex64::new(inv_sqrt_two, 0.0),
        ]),
        frame(&[
            Complex64::new(inv_sqrt_two, 0.0),
            Complex64::new(0.0, inv_sqrt_two),
        ]),
        frame(&[
            Complex64::new(inv_sqrt_two, 0.0),
            Complex64::new(inv_sqrt_two, 0.0),
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
    let transformed_phase = wilson_line_phase(&transformed).unwrap();
    assert!((transformed_phase - phase).abs() < 1.0e-12);
}

#[test]
fn constant_plaquette_has_zero_flux() {
    let state = frame(&[Complex64::new(1.0, 0.0)]);
    let flux = plaquette_flux(&[state.clone(), state.clone(), state.clone(), state]).unwrap();
    assert_eq!(flux, 0.0);
}

#[test]
fn parallel_transport_link_is_unitary_for_rotated_frames() {
    let inv_sqrt_two = 1.0 / 2.0_f64.sqrt();
    let left = ComplexMatrix::identity(2);
    let right = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(inv_sqrt_two, 0.0),
            Complex64::new(inv_sqrt_two, 0.0),
            Complex64::new(-inv_sqrt_two, 0.0),
            Complex64::new(inv_sqrt_two, 0.0),
        ],
    )
    .unwrap();
    let link = parallel_transport_link(&left, &right).unwrap();

    for row in 0..2 {
        for column in 0..2 {
            let product: Complex64 = (0..2)
                .map(|inner| {
                    link.get(row, inner).unwrap() * link.get(column, inner).unwrap().conj()
                })
                .sum();
            let expected = if row == column { 1.0 } else { 0.0 };
            assert!((product - Complex64::new(expected, 0.0)).norm() < 1.0e-12);
        }
    }
}

#[test]
fn wilson_eigenphases_resolve_multiband_transport() {
    let angle = 0.4;
    let first = ComplexMatrix::identity(2);
    let second = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::from_polar(1.0, angle),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::from_polar(1.0, -angle),
        ],
    )
    .unwrap();
    let phases = wilson_loop_eigenphases(&[first, second]).unwrap();
    assert!((phases[0] + angle).abs() < 1.0e-12);
    assert!((phases[1] - angle).abs() < 1.0e-12);
}

#[test]
fn unitary_link_logarithm_produces_hermitian_connection() {
    let link = ComplexMatrix::scalar(Complex64::from_polar(1.0, 0.2));
    let connection = connection_from_link(&link, 0.1).unwrap();
    assert!((connection.get(0, 0).unwrap() - Complex64::new(-2.0, 0.0)).norm() < 1.0e-12);
    assert!(connection.is_hermitian(1.0e-12).unwrap());
}
