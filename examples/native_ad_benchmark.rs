use std::time::{Duration, Instant};

use thouless::ad::{
    AffineHermitianFamily, KpmMomentObjective, ModelDirection, ModelParameters,
    SparseAffineOperator, SparseHermitianTerm, SpectralProjectorObjective,
};
use thouless::linear_operator::CsrMatrix;
use thouless::{Complex64, ComplexMatrix};

const FINITE_DIFFERENCE_STEP: f64 = 1.0e-6;

fn elapsed_per_call(mut operation: impl FnMut(), repetitions: usize) -> Duration {
    let started = Instant::now();
    for _ in 0..repetitions {
        operation();
    }
    started.elapsed() / repetitions as u32
}

fn maximum_relative_error(reference: &[f64], candidate: &[f64]) -> f64 {
    reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| {
            (reference - candidate).abs() / reference.abs().max(candidate.abs()).max(1.0)
        })
        .fold(0.0, f64::max)
}

fn projector_benchmark(parameter_count: usize) {
    let dimension = 18;
    let occupied = dimension / 2;
    let mut base = ComplexMatrix::zeros(dimension, dimension);
    for index in 0..dimension {
        base.set(
            index,
            index,
            Complex64::new(index as f64 - occupied as f64 + 0.4, 0.0),
        )
        .unwrap();
        if index + 1 < dimension {
            let hopping = Complex64::new(0.035, 0.012);
            base.set(index, index + 1, hopping).unwrap();
            base.set(index + 1, index, hopping.conj()).unwrap();
        }
    }
    let directions = (0..parameter_count)
        .map(|parameter| {
            let mut direction = ComplexMatrix::zeros(dimension, dimension);
            let first = parameter % dimension;
            let mut second = (7 * parameter + 3) % dimension;
            if second == first {
                second = (second + 1) % dimension;
            }
            direction
                .set(
                    first,
                    first,
                    Complex64::new(0.025 * (1 + parameter % 5) as f64, 0.0),
                )
                .unwrap();
            let hopping = Complex64::new(
                0.008 * ((parameter % 3) as f64 - 1.0),
                0.006 * ((parameter % 4) as f64 - 1.5),
            );
            direction.set(first, second, hopping).unwrap();
            direction.set(second, first, hopping.conj()).unwrap();
            direction
        })
        .collect::<Vec<_>>();
    let family = AffineHermitianFamily::new(base, directions).unwrap();
    let mut target = ComplexMatrix::zeros(dimension, dimension);
    for index in 0..occupied {
        target.set(index, index, Complex64::new(1.0, 0.0)).unwrap();
    }
    let objective = SpectralProjectorObjective::new(&family, occupied, target, 1.0e-4).unwrap();
    let parameters = ModelParameters::new(
        (0..parameter_count)
            .map(|index| 0.03 * (index as f64 + 0.5).sin())
            .collect(),
    )
    .unwrap();

    let (_, native_gradient) = objective.value_and_grad(&parameters).unwrap();
    let finite_difference = (0..parameter_count)
        .map(|parameter| {
            let mut coordinate = vec![0.0; parameter_count];
            coordinate[parameter] = 1.0;
            let direction = ModelDirection::new(coordinate).unwrap();
            let positive = objective
                .value(
                    &parameters
                        .displaced(&direction, FINITE_DIFFERENCE_STEP)
                        .unwrap(),
                )
                .unwrap();
            let negative = objective
                .value(
                    &parameters
                        .displaced(&direction, -FINITE_DIFFERENCE_STEP)
                        .unwrap(),
                )
                .unwrap();
            (positive - negative) / (2.0 * FINITE_DIFFERENCE_STEP)
        })
        .collect::<Vec<_>>();
    let error = maximum_relative_error(native_gradient.as_slice(), &finite_difference);
    assert!(error < 2.0e-7, "projector gradient error {error:e}");

    let native_time = elapsed_per_call(
        || {
            std::hint::black_box(objective.value_and_grad(&parameters).unwrap());
        },
        6,
    );
    let finite_difference_time = elapsed_per_call(
        || {
            for parameter in 0..parameter_count {
                let mut coordinate = vec![0.0; parameter_count];
                coordinate[parameter] = 1.0;
                let direction = ModelDirection::new(coordinate).unwrap();
                std::hint::black_box(
                    objective
                        .value(
                            &parameters
                                .displaced(&direction, FINITE_DIFFERENCE_STEP)
                                .unwrap(),
                        )
                        .unwrap(),
                );
                std::hint::black_box(
                    objective
                        .value(
                            &parameters
                                .displaced(&direction, -FINITE_DIFFERENCE_STEP)
                                .unwrap(),
                        )
                        .unwrap(),
                );
            }
        },
        2,
    );
    println!(
        concat!(
            "{{\"benchmark\":\"spectral_projector\",",
            "\"parameter_count\":{},\"dimension\":{},",
            "\"native_microseconds\":{},\"finite_difference_microseconds\":{},",
            "\"speedup\":{:.6},\"native_eigensystems\":1,",
            "\"finite_difference_eigensystems\":{},\"maximum_relative_error\":{:.6e}}}"
        ),
        parameter_count,
        dimension,
        native_time.as_micros(),
        finite_difference_time.as_micros(),
        finite_difference_time.as_secs_f64() / native_time.as_secs_f64(),
        2 * parameter_count,
        error,
    );
}

fn sparse_chain(dimension: usize) -> CsrMatrix {
    let mut row_offsets = Vec::with_capacity(dimension + 1);
    let mut column_indices = Vec::with_capacity(3 * dimension);
    let mut values = Vec::with_capacity(3 * dimension);
    row_offsets.push(0);
    for row in 0..dimension {
        if row > 0 {
            column_indices.push(row - 1);
            values.push(Complex64::new(0.21, -0.015));
        }
        column_indices.push(row);
        values.push(Complex64::new(0.04 * (0.1 * row as f64).cos(), 0.0));
        if row + 1 < dimension {
            column_indices.push(row + 1);
            values.push(Complex64::new(0.21, 0.015));
        }
        row_offsets.push(column_indices.len());
    }
    CsrMatrix::new(dimension, dimension, row_offsets, column_indices, values).unwrap()
}

fn kpm_benchmark(parameter_count: usize) {
    let dimension = 384;
    let moment_count = 72;
    let family = SparseAffineOperator::new(
        sparse_chain(dimension),
        parameter_count,
        (0..parameter_count)
            .map(|parameter| SparseHermitianTerm {
                parameter,
                row: (13 * parameter + 7) % dimension,
                column: (13 * parameter + 7) % dimension,
                coefficient: Complex64::new(0.03, 0.0),
            })
            .collect(),
    )
    .unwrap();
    let normalization = (dimension as f64).sqrt();
    let probe = (0..dimension)
        .map(|index| {
            Complex64::new(
                (0.37 * index as f64).cos() / normalization,
                (0.19 * index as f64).sin() / normalization,
            )
        })
        .collect();
    let coefficients = (0..moment_count)
        .map(|moment| (-0.035 * moment as f64).exp() * (0.23 * moment as f64).cos())
        .collect();
    let objective = KpmMomentObjective::new(&family, probe, coefficients, 9).unwrap();
    let parameters = ModelParameters::new(
        (0..parameter_count)
            .map(|index| 0.05 * (0.4 * index as f64).sin())
            .collect(),
    )
    .unwrap();

    let report = objective.value_and_grad_with_report(&parameters).unwrap();
    let finite_difference = (0..parameter_count)
        .map(|parameter| {
            let mut coordinate = vec![0.0; parameter_count];
            coordinate[parameter] = 1.0;
            let direction = ModelDirection::new(coordinate).unwrap();
            let positive = objective
                .value(
                    &parameters
                        .displaced(&direction, FINITE_DIFFERENCE_STEP)
                        .unwrap(),
                )
                .unwrap();
            let negative = objective
                .value(
                    &parameters
                        .displaced(&direction, -FINITE_DIFFERENCE_STEP)
                        .unwrap(),
                )
                .unwrap();
            (positive - negative) / (2.0 * FINITE_DIFFERENCE_STEP)
        })
        .collect::<Vec<_>>();
    let error = maximum_relative_error(report.gradient().as_slice(), &finite_difference);
    assert!(error < 2.0e-7, "KPM gradient error {error:e}");

    let native_time = elapsed_per_call(
        || {
            std::hint::black_box(objective.value_and_grad_with_report(&parameters).unwrap());
        },
        5,
    );
    let finite_difference_time = elapsed_per_call(
        || {
            for parameter in 0..parameter_count {
                let mut coordinate = vec![0.0; parameter_count];
                coordinate[parameter] = 1.0;
                let direction = ModelDirection::new(coordinate).unwrap();
                std::hint::black_box(
                    objective
                        .value(
                            &parameters
                                .displaced(&direction, FINITE_DIFFERENCE_STEP)
                                .unwrap(),
                        )
                        .unwrap(),
                );
                std::hint::black_box(
                    objective
                        .value(
                            &parameters
                                .displaced(&direction, -FINITE_DIFFERENCE_STEP)
                                .unwrap(),
                        )
                        .unwrap(),
                );
            }
        },
        2,
    );
    let native_operator_applications = report.forward_operator_applications()
        + report.recomputed_operator_applications()
        + report.adjoint_operator_applications();
    let finite_difference_operator_applications =
        2 * parameter_count * report.forward_operator_applications();
    println!(
        concat!(
            "{{\"benchmark\":\"sparse_kpm\",",
            "\"parameter_count\":{},\"dimension\":{},\"moment_count\":{},",
            "\"native_microseconds\":{},\"finite_difference_microseconds\":{},",
            "\"speedup\":{:.6},\"native_operator_applications\":{},",
            "\"finite_difference_operator_applications\":{},",
            "\"peak_stored_vectors\":{},\"full_tape_vectors\":{},",
            "\"maximum_relative_error\":{:.6e}}}"
        ),
        parameter_count,
        dimension,
        moment_count,
        native_time.as_micros(),
        finite_difference_time.as_micros(),
        finite_difference_time.as_secs_f64() / native_time.as_secs_f64(),
        native_operator_applications,
        finite_difference_operator_applications,
        report.peak_stored_vectors(),
        moment_count,
        error,
    );
}

fn main() {
    projector_benchmark(48);
    kpm_benchmark(48);
}
