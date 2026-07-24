use thouless::topology::{plaquette_flux, wilson_line_phase};
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
