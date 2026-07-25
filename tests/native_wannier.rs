use std::f64::consts::TAU;

use thouless::wannier::{
    disentangle_subspace, interpolate_periodic_matrices, maximize_localization, periodic_overlaps,
    project_trials, spread_decomposition,
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

#[test]
fn two_dimensional_composite_gauge_localizes_without_changing_the_subspace() {
    let shape = [7, 6];
    let frames = (0..shape[0])
        .flat_map(|first| {
            (0..shape[1]).map(move |second| {
                let kx = TAU * first as f64 / shape[0] as f64;
                let ky = TAU * second as f64 / shape[1] as f64;
                let angle = 0.41 * kx.sin() + 0.27 * ky.cos();
                let first_phase = Complex64::from_polar(1.0, 0.22 * ky.sin());
                let second_phase = Complex64::from_polar(1.0, -0.19 * kx.cos());
                matrix(
                    2,
                    2,
                    vec![
                        first_phase * angle.cos(),
                        first_phase * angle.sin(),
                        -second_phase * angle.sin(),
                        second_phase * angle.cos(),
                    ],
                )
            })
        })
        .collect::<Vec<_>>();
    let step_x = TAU / shape[0] as f64;
    let step_y = TAU / shape[1] as f64;
    let vectors = vec![
        vec![step_x, 0.0],
        vec![-step_x, 0.0],
        vec![0.0, step_y],
        vec![0.0, -step_y],
    ];
    let weights = vec![
        1.0 / (2.0 * step_x.powi(2)),
        1.0 / (2.0 * step_x.powi(2)),
        1.0 / (2.0 * step_y.powi(2)),
        1.0 / (2.0 * step_y.powi(2)),
    ];

    let report = maximize_localization(
        &shape,
        &frames,
        &[vec![1, 0], vec![-1, 0], vec![0, 1], vec![0, -1]],
        &[
            vec![Complex64::new(1.0, 0.0); 2],
            vec![Complex64::new(1.0, 0.0); 2],
        ],
        &vectors,
        &weights,
        0.5,
        300,
        1.0e-12,
        1.0e-9,
    )
    .unwrap();

    assert!(report.iterations() > 0);
    assert!(report.final_spread() < report.initial_spread() * 1.0e-5);
    for frame in report.frames() {
        for left in 0..2 {
            for right in 0..2 {
                let overlap = (0..2)
                    .map(|basis| {
                        frame.as_slice()[left * 2 + basis].conj()
                            * frame.as_slice()[right * 2 + basis]
                    })
                    .sum::<Complex64>();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((overlap - Complex64::new(expected, 0.0)).norm() < 1.0e-10);
            }
        }
    }
}

#[test]
fn disentanglement_preserves_the_frozen_manifold_exactly() {
    let sample_count = 13;
    let candidates = vec![ComplexMatrix::identity(3); sample_count];
    let initial = (0..sample_count)
        .map(|sample| {
            let momentum = TAU * sample as f64 / sample_count as f64;
            let angle = 0.52 * momentum.sin();
            matrix(
                2,
                3,
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(angle.cos(), 0.0),
                    Complex64::new(angle.sin(), 0.0),
                ],
            )
        })
        .collect::<Vec<_>>();

    let report = disentangle_subspace(
        &[sample_count],
        &candidates,
        &vec![1; sample_count],
        2,
        Some(&initial),
        None,
        &[vec![1], vec![-1]],
        &[vec![Complex64::new(1.0, 0.0); 3]],
        &[1.0, 1.0],
        200,
        1.0e-12,
        0.8,
    )
    .unwrap();

    assert!(report.final_invariant_spread() <= report.initial_invariant_spread());
    for frame in report.frames() {
        assert_eq!(
            &frame.as_slice()[..3],
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
    }
}
