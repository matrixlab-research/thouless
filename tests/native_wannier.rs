use std::f64::consts::TAU;

use thouless::wannier::{
    interpolate_periodic_matrices, periodic_overlaps, project_trials, spread_decomposition,
};
use thouless::{Complex64, ComplexMatrix};

fn matrix(rows: usize, columns: usize, entries: Vec<Complex64>) -> ComplexMatrix {
    ComplexMatrix::new(rows, columns, entries).unwrap()
}

#[test]
fn trial_projection_is_independent_of_the_source_frame_gauge() {
    let inverse_sqrt_two = 1.0 / 2.0f64.sqrt();
    let frames = (0..7)
        .map(|sample| {
            let phase = Complex64::from_polar(1.0, 0.31 * sample as f64);
            matrix(
                2,
                2,
                vec![
                    phase * inverse_sqrt_two,
                    Complex64::new(inverse_sqrt_two, 0.0),
                    -phase * inverse_sqrt_two,
                    Complex64::new(inverse_sqrt_two, 0.0),
                ],
            )
        })
        .collect::<Vec<_>>();
    let trial = matrix(
        1,
        2,
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    );

    let projected = project_trials(&frames, &trial, 1.0e-12).unwrap();

    for frame in projected {
        assert!(frame.as_slice()[0].norm() < 1.0e-12);
        assert!((frame.as_slice()[1] - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
    }
}

#[test]
fn embedded_atomic_orbital_has_zero_quadratic_spread() {
    let sample_count = 11;
    let embedding = 0.37;
    let frames = (0..sample_count)
        .map(|sample| {
            matrix(
                1,
                1,
                vec![Complex64::from_polar(
                    1.0,
                    -TAU * embedding * sample as f64 / sample_count as f64,
                )],
            )
        })
        .collect::<Vec<_>>();
    let overlaps = periodic_overlaps(
        &[sample_count],
        &frames,
        &[vec![1], vec![-1]],
        &[vec![Complex64::from_polar(1.0, -TAU * embedding)]],
    )
    .unwrap();
    let step = TAU / sample_count as f64;
    let weight = 1.0 / (2.0 * step * step);

    let spread =
        spread_decomposition(&overlaps, &[vec![step], vec![-step]], &[weight, weight]).unwrap();

    assert!((spread.centers()[0][0] - embedding).abs() < 1.0e-12);
    assert!(spread.spreads()[0].abs() < 1.0e-12);
    assert!(spread.invariant().abs() < 1.0e-12);
    assert!(spread.diagonal().abs() < 1.0e-12);
    assert!(spread.off_diagonal().abs() < 1.0e-12);
}

#[test]
fn two_dimensional_matrix_interpolation_generalizes_off_mesh() {
    let shape = [6, 5];
    let samples = (0..shape[0])
        .flat_map(|first| {
            (0..shape[1]).map(move |second| {
                let kx = first as f64 / shape[0] as f64;
                let ky = second as f64 / shape[1] as f64;
                let diagonal = 2.0 * (TAU * kx).cos();
                let off_diagonal = (TAU * ky).sin();
                matrix(
                    2,
                    2,
                    vec![
                        Complex64::new(diagonal, 0.0),
                        Complex64::new(off_diagonal, 0.0),
                        Complex64::new(off_diagonal, 0.0),
                        Complex64::new(-diagonal, 0.0),
                    ],
                )
            })
        })
        .collect::<Vec<_>>();
    let points = vec![vec![0.137, 0.281], vec![0.413, -0.177]];

    let interpolated = interpolate_periodic_matrices(&shape, &samples, &points).unwrap();

    for (matrix, point) in interpolated.iter().zip(&points) {
        let diagonal = 2.0 * (TAU * point[0]).cos();
        let off_diagonal = (TAU * point[1]).sin();
        assert!((matrix.as_slice()[0].re - diagonal).abs() < 1.0e-12);
        assert!((matrix.as_slice()[1].re - off_diagonal).abs() < 1.0e-12);
        assert!((matrix.as_slice()[2].re - off_diagonal).abs() < 1.0e-12);
        assert!((matrix.as_slice()[3].re + diagonal).abs() < 1.0e-12);
        assert!(matrix
            .as_slice()
            .iter()
            .all(|value| value.im.abs() < 1.0e-12));
    }
}
