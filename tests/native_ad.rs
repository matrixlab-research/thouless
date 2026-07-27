use chainrules_core::{JvpRule, Pullback, VjpRule};
use thouless::ad::{
    real_frobenius_pairing, AdError, AffineHermitianFamily, DifferentiableLead,
    DifferentiableOpenSystem, IsolatedEigenvalue, KpmMomentObjective, LeadDirection, LinearSolve,
    LinearSolveArguments, LinearSolveTangent, ModelDirection, ModelParameters, OpenSystemDirection,
    OpenSystemTransmission, OpenTransmissionObjective, ProjectorDistance,
    QuantumMetricMeshObjective, SparseAffineOperator, SparseHermitianTerm,
    SparseLinearFunctionalObjective, SpectralProjectorObjective, SurfaceGreenArguments,
    SurfaceGreenRule, SurfaceGreenTangent,
};
use thouless::linear_operator::{AdjointLinearOperator, CsrMatrix, GmresOptions, LinearOperator};
use thouless::spectrum::hermitian_eigensystem;
use thouless::{Complex64, ComplexMatrix};

const FD_STEP: f64 = 1.0e-6;
const ABSOLUTE_TOLERANCE: f64 = 2.0e-7;
const RELATIVE_TOLERANCE: f64 = 2.0e-6;

fn matrix(rows: usize, columns: usize, values: &[(f64, f64)]) -> ComplexMatrix {
    ComplexMatrix::new(
        rows,
        columns,
        values
            .iter()
            .map(|&(real, imaginary)| Complex64::new(real, imaginary))
            .collect(),
    )
    .unwrap()
}

fn affine_two_level() -> AffineHermitianFamily {
    let base = matrix(2, 2, &[(-0.8, 0.0), (0.25, -0.1), (0.25, 0.1), (0.9, 0.0)]);
    let sigma_z = matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)]);
    let sigma_x = matrix(2, 2, &[(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]);
    let sigma_y = matrix(2, 2, &[(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)]);
    AffineHermitianFamily::new(base, vec![sigma_z, sigma_x, sigma_y]).unwrap()
}

fn occupied_projector(hamiltonian: &ComplexMatrix, occupied: usize) -> ComplexMatrix {
    let eigen = hermitian_eigensystem(hamiltonian, 1.0e-12).unwrap();
    let dimension = hamiltonian.rows();
    let mut projector = ComplexMatrix::zeros(dimension, dimension);
    for state in 0..occupied {
        for row in 0..dimension {
            for column in 0..dimension {
                let left = eigen.eigenvectors().as_slice()[row * dimension + state];
                let right = eigen.eigenvectors().as_slice()[column * dimension + state];
                projector
                    .add_entry(row, column, left * right.conj())
                    .unwrap();
            }
        }
    }
    projector
}

fn real_vector_pairing(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left.conj() * right).re)
        .sum()
}

fn assert_close(actual: f64, expected: f64) {
    chainrules_core::testing::assert_close(
        actual,
        expected,
        RELATIVE_TOLERANCE,
        ABSOLUTE_TOLERANCE,
    );
}

#[test]
fn affine_hamiltonian_jvp_and_vjp_are_dual_and_match_finite_difference() {
    let family = affine_two_level();
    let parameters = ModelParameters::new(vec![0.17, -0.12, 0.08]).unwrap();
    let direction = ModelDirection::new(vec![0.4, -0.3, 0.2]).unwrap();
    let matrix_cotangent = matrix(2, 2, &[(0.2, 0.0), (-0.4, 0.3), (-0.4, -0.3), (0.1, 0.0)]);

    let (_, matrix_tangent) = family.jvp(&parameters, &direction).unwrap();
    let (_, pullback) = family.vjp(&parameters).unwrap();
    let parameter_cotangent = pullback.apply(matrix_cotangent.clone()).unwrap();

    chainrules_core::testing::assert_adjoint_identity(
        &matrix_tangent,
        &matrix_cotangent,
        &direction,
        &parameter_cotangent,
        |left, right| real_frobenius_pairing(left, right).unwrap(),
        |left, right| {
            left.as_slice()
                .iter()
                .zip(right.as_slice())
                .map(|(left, right)| left * right)
                .sum()
        },
        1.0e-12,
        1.0e-12,
    );

    let positive = family
        .value(&parameters.displaced(&direction, FD_STEP).unwrap())
        .unwrap();
    let negative = family
        .value(&parameters.displaced(&direction, -FD_STEP).unwrap())
        .unwrap();
    for ((positive, negative), analytic) in positive
        .as_slice()
        .iter()
        .zip(negative.as_slice())
        .zip(matrix_tangent.as_slice())
    {
        assert_close(((positive - negative) / (2.0 * FD_STEP)).re, analytic.re);
        assert_close(((positive - negative) / (2.0 * FD_STEP)).im, analytic.im);
    }
}

#[test]
fn dense_linear_solve_rule_matches_finite_difference_and_adjoint_identity() {
    let arguments = LinearSolveArguments {
        matrix: matrix(2, 2, &[(2.0, 0.2), (-0.3, 0.4), (0.1, -0.2), (1.7, -0.1)]),
        right_hand_side: vec![Complex64::new(0.7, -0.1), Complex64::new(-0.2, 0.5)],
    };
    let tangent = LinearSolveTangent {
        matrix: matrix(
            2,
            2,
            &[(0.2, -0.1), (0.05, 0.08), (-0.04, 0.03), (-0.1, 0.02)],
        ),
        right_hand_side: vec![Complex64::new(0.1, 0.02), Complex64::new(-0.03, 0.06)],
    };
    let output_cotangent = vec![Complex64::new(0.4, -0.2), Complex64::new(-0.1, 0.3)];

    let (_, solution_tangent) = LinearSolve.jvp(&arguments, &tangent).unwrap();
    let (_, pullback) = LinearSolve.vjp(&arguments).unwrap();
    let input_cotangent = pullback.apply(output_cotangent.clone()).unwrap();

    let output_pairing = real_vector_pairing(&solution_tangent, &output_cotangent);
    let input_pairing = real_frobenius_pairing(&tangent.matrix, &input_cotangent.matrix).unwrap()
        + real_vector_pairing(&tangent.right_hand_side, &input_cotangent.right_hand_side);
    assert_close(output_pairing, input_pairing);

    let displaced = |scale: f64| LinearSolveArguments {
        matrix: ComplexMatrix::new(
            2,
            2,
            arguments
                .matrix
                .as_slice()
                .iter()
                .zip(tangent.matrix.as_slice())
                .map(|(value, direction)| value + direction * scale)
                .collect(),
        )
        .unwrap(),
        right_hand_side: arguments
            .right_hand_side
            .iter()
            .zip(&tangent.right_hand_side)
            .map(|(value, direction)| value + direction * scale)
            .collect(),
    };
    let (positive, _) = LinearSolve.vjp(&displaced(FD_STEP)).unwrap();
    let (negative, _) = LinearSolve.vjp(&displaced(-FD_STEP)).unwrap();
    for ((positive, negative), analytic) in positive.iter().zip(negative).zip(&solution_tangent) {
        let numerical = (positive - negative) / (2.0 * FD_STEP);
        assert_close(numerical.re, analytic.re);
        assert_close(numerical.im, analytic.im);
    }
}

#[test]
fn isolated_eigenvalue_rule_matches_feynman_hellmann_and_finite_difference() {
    let family = affine_two_level();
    let parameters = ModelParameters::new(vec![0.11, 0.07, -0.04]).unwrap();
    let direction = ModelDirection::new(vec![0.3, -0.2, 0.5]).unwrap();
    let (hamiltonian, hamiltonian_tangent) = family.jvp(&parameters, &direction).unwrap();
    let rule = IsolatedEigenvalue::new(0, 1.0e-4).unwrap();
    let (_, analytic) = rule.jvp(&hamiltonian, &hamiltonian_tangent).unwrap();

    let positive = rule
        .vjp(
            &family
                .value(&parameters.displaced(&direction, FD_STEP).unwrap())
                .unwrap(),
        )
        .unwrap()
        .0;
    let negative = rule
        .vjp(
            &family
                .value(&parameters.displaced(&direction, -FD_STEP).unwrap())
                .unwrap(),
        )
        .unwrap()
        .0;
    assert_close((positive - negative) / (2.0 * FD_STEP), analytic);
}

#[test]
fn projector_rule_is_gauge_invariant_and_matches_directional_difference() {
    let family = affine_two_level();
    let target_parameters = ModelParameters::new(vec![-0.2, 0.15, 0.1]).unwrap();
    let target = occupied_projector(&family.value(&target_parameters).unwrap(), 1);
    let objective = SpectralProjectorObjective::new(&family, 1, target, 1.0e-4).unwrap();
    let parameters = ModelParameters::new(vec![0.1, -0.05, 0.08]).unwrap();
    let direction = ModelDirection::new(vec![0.2, 0.4, -0.3]).unwrap();

    let (value, analytic) = objective.jvp(&parameters, &direction).unwrap();
    assert!(value > 0.0);
    let positive = objective
        .value(&parameters.displaced(&direction, FD_STEP).unwrap())
        .unwrap();
    let negative = objective
        .value(&parameters.displaced(&direction, -FD_STEP).unwrap())
        .unwrap();
    assert_close((positive - negative) / (2.0 * FD_STEP), analytic);

    let (value, gradient) = objective.value_and_grad(&parameters).unwrap();
    let dot = gradient
        .as_slice()
        .iter()
        .zip(direction.as_slice())
        .map(|(gradient, direction)| gradient * direction)
        .sum::<f64>();
    assert_close(dot, analytic);
    assert!(value.is_finite());

    let hamiltonian = family.value(&parameters).unwrap();
    let projector = occupied_projector(&hamiltonian, 1);
    let direct = ProjectorDistance::new(1, projector, 1.0e-4).unwrap();
    assert_close(direct.value_and_gradient(&hamiltonian).unwrap().0, 0.0);
}

#[test]
fn transmission_vjp_matches_finite_difference_for_many_parameters() {
    let family = affine_two_level();
    let left = matrix(2, 2, &[(0.0, -0.15), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    let right = matrix(2, 2, &[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, -0.2)]);
    let objective = OpenTransmissionObjective::new(&family, vec![left, right], 0.13, 1, 0).unwrap();
    let parameters = ModelParameters::new(vec![0.08, -0.06, 0.03]).unwrap();
    let direction = ModelDirection::new(vec![0.4, -0.2, 0.3]).unwrap();

    let (_, analytic) = objective.jvp(&parameters, &direction).unwrap();
    let positive = objective
        .value(&parameters.displaced(&direction, FD_STEP).unwrap())
        .unwrap();
    let negative = objective
        .value(&parameters.displaced(&direction, -FD_STEP).unwrap())
        .unwrap();
    assert_close((positive - negative) / (2.0 * FD_STEP), analytic);

    let (_, gradient) = objective.value_and_grad(&parameters).unwrap();
    let dot = gradient
        .as_slice()
        .iter()
        .zip(direction.as_slice())
        .map(|(gradient, direction)| gradient * direction)
        .sum::<f64>();
    assert_close(dot, analytic);
}

#[test]
fn sparse_parameterized_operator_preserves_adjoint_and_parameter_duality() {
    let base = CsrMatrix::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(-0.4, 0.1),
            Complex64::new(-0.4, -0.1),
            Complex64::new(0.2, 0.0),
            Complex64::new(-0.3, 0.0),
            Complex64::new(-0.3, 0.0),
            Complex64::new(-0.1, 0.0),
        ],
    )
    .unwrap();
    let family = SparseAffineOperator::new(
        base,
        3,
        vec![
            SparseHermitianTerm {
                parameter: 0,
                row: 0,
                column: 0,
                coefficient: Complex64::new(1.0, 0.0),
            },
            SparseHermitianTerm {
                parameter: 1,
                row: 1,
                column: 1,
                coefficient: Complex64::new(1.0, 0.0),
            },
            SparseHermitianTerm {
                parameter: 2,
                row: 2,
                column: 2,
                coefficient: Complex64::new(1.0, 0.0),
            },
        ],
    )
    .unwrap();
    let parameters = ModelParameters::new(vec![0.1, -0.2, 0.3]).unwrap();
    let direction = ModelDirection::new(vec![0.4, 0.2, -0.1]).unwrap();
    let input = vec![
        Complex64::new(0.3, -0.2),
        Complex64::new(-0.1, 0.4),
        Complex64::new(0.2, 0.1),
    ];
    let output_cotangent = vec![
        Complex64::new(-0.2, 0.1),
        Complex64::new(0.5, -0.3),
        Complex64::new(0.1, 0.2),
    ];

    let bound = family.bind(&parameters).unwrap();
    let output = bound.apply(&input).unwrap();
    let adjoint = bound.apply_adjoint(&output_cotangent).unwrap();
    assert_close(
        real_vector_pairing(&output, &output_cotangent),
        real_vector_pairing(&input, &adjoint),
    );

    let mut parameter_jvp = vec![Complex64::new(0.0, 0.0); 3];
    family
        .parameter_jvp_into(&direction, &input, &mut parameter_jvp)
        .unwrap();
    let gradient = family.parameter_vjp(&input, &output_cotangent).unwrap();
    let left = real_vector_pairing(&parameter_jvp, &output_cotangent);
    let right = direction
        .as_slice()
        .iter()
        .zip(gradient.as_slice())
        .map(|(direction, gradient)| direction * gradient)
        .sum();
    assert_close(left, right);
}

#[test]
fn sparse_solve_uses_one_primal_and_one_adjoint_system_for_many_parameters() {
    let base = CsrMatrix::new(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            Complex64::new(2.4, 0.0),
            Complex64::new(-0.3, 0.08),
            Complex64::new(-0.3, -0.08),
            Complex64::new(2.1, 0.0),
            Complex64::new(-0.25, 0.0),
            Complex64::new(-0.25, 0.0),
            Complex64::new(2.3, 0.0),
            Complex64::new(-0.2, -0.04),
            Complex64::new(-0.2, 0.04),
            Complex64::new(1.9, 0.0),
        ],
    )
    .unwrap();
    let family = SparseAffineOperator::new(
        base,
        4,
        (0..4)
            .map(|parameter| SparseHermitianTerm {
                parameter,
                row: parameter,
                column: parameter,
                coefficient: Complex64::new(0.2, 0.0),
            })
            .collect(),
    )
    .unwrap();
    let objective = SparseLinearFunctionalObjective::new(
        &family,
        vec![
            Complex64::new(1.0, 0.2),
            Complex64::new(-0.3, 0.1),
            Complex64::new(0.5, -0.2),
            Complex64::new(0.1, 0.4),
        ],
        vec![
            Complex64::new(0.2, -0.1),
            Complex64::new(0.4, 0.3),
            Complex64::new(-0.2, 0.2),
            Complex64::new(0.1, -0.3),
        ],
        GmresOptions {
            relative_tolerance: 1.0e-13,
            absolute_tolerance: 1.0e-14,
            restart: 4,
            max_iterations: 32,
        },
    )
    .unwrap();
    let parameters = ModelParameters::new(vec![0.1, -0.2, 0.05, 0.12]).unwrap();
    let direction = ModelDirection::new(vec![0.3, -0.1, 0.2, 0.4]).unwrap();

    let (_, jvp) = objective.jvp(&parameters, &direction).unwrap();
    let report = objective.value_and_grad_with_report(&parameters).unwrap();
    let vjp = report
        .gradient()
        .as_slice()
        .iter()
        .zip(direction.as_slice())
        .map(|(gradient, direction)| gradient * direction)
        .sum::<f64>();
    assert_close(vjp, jvp);
    let positive = objective
        .value(&parameters.displaced(&direction, FD_STEP).unwrap())
        .unwrap();
    let negative = objective
        .value(&parameters.displaced(&direction, -FD_STEP).unwrap())
        .unwrap();
    assert_close((positive - negative) / (2.0 * FD_STEP), jvp);
    assert!(report.primal_iterations() <= 4);
    assert!(report.adjoint_iterations() <= 4);
    assert!(report.primal_residual_norm() < 1.0e-12);
    assert!(report.adjoint_residual_norm() < 1.0e-12);
}

#[test]
fn checkpointed_kpm_reverse_matches_jvp_and_finite_difference() {
    let base = CsrMatrix::new(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            Complex64::new(-0.2, 0.0),
            Complex64::new(0.17, 0.04),
            Complex64::new(0.17, -0.04),
            Complex64::new(0.1, 0.0),
            Complex64::new(-0.13, 0.02),
            Complex64::new(-0.13, -0.02),
            Complex64::new(0.05, 0.0),
            Complex64::new(0.11, -0.03),
            Complex64::new(0.11, 0.03),
            Complex64::new(-0.08, 0.0),
        ],
    )
    .unwrap();
    let family = SparseAffineOperator::new(
        base,
        3,
        vec![
            SparseHermitianTerm {
                parameter: 0,
                row: 0,
                column: 0,
                coefficient: Complex64::new(0.3, 0.0),
            },
            SparseHermitianTerm {
                parameter: 1,
                row: 1,
                column: 2,
                coefficient: Complex64::new(0.12, -0.07),
            },
            SparseHermitianTerm {
                parameter: 2,
                row: 3,
                column: 3,
                coefficient: Complex64::new(-0.25, 0.0),
            },
        ],
    )
    .unwrap();
    let probe = vec![
        Complex64::new(0.5, 0.1),
        Complex64::new(-0.2, 0.3),
        Complex64::new(0.4, -0.15),
        Complex64::new(0.1, 0.25),
    ];
    let coefficients = vec![0.2, -0.4, 0.1, 0.25, -0.08, 0.03, 0.07, -0.02, 0.01];
    let objective = KpmMomentObjective::new(&family, probe, coefficients, 3).unwrap();
    let parameters = ModelParameters::new(vec![0.08, -0.04, 0.11]).unwrap();
    let direction = ModelDirection::new(vec![0.3, -0.2, 0.4]).unwrap();

    let (_, directional) = objective.jvp(&parameters, &direction).unwrap();
    let report = objective.value_and_grad_with_report(&parameters).unwrap();
    let reverse_directional = report
        .gradient()
        .as_slice()
        .iter()
        .zip(direction.as_slice())
        .map(|(gradient, direction)| gradient * direction)
        .sum::<f64>();
    assert_close(reverse_directional, directional);

    let positive = objective
        .value(&parameters.displaced(&direction, FD_STEP).unwrap())
        .unwrap();
    let negative = objective
        .value(&parameters.displaced(&direction, -FD_STEP).unwrap())
        .unwrap();
    assert_close((positive - negative) / (2.0 * FD_STEP), directional);

    assert_eq!(
        report.forward_operator_applications(),
        objective.moment_count() - 1
    );
    assert_eq!(
        report.adjoint_operator_applications(),
        objective.moment_count() - 2
    );
    assert!(
        report.peak_stored_vectors() < objective.moment_count() + report.checkpoint_count(),
        "checkpointing should retain fewer state vectors than a tape of every moment"
    );
}

#[test]
fn surface_green_implicit_rule_matches_finite_difference_and_adjoint_identity() {
    let arguments = SurfaceGreenArguments {
        cell_hamiltonian: matrix(
            2,
            2,
            &[(0.15, 0.0), (-0.08, 0.03), (-0.08, -0.03), (-0.12, 0.0)],
        ),
        inter_cell_hopping: matrix(
            2,
            2,
            &[(0.31, 0.02), (0.04, -0.03), (-0.02, 0.01), (0.27, -0.01)],
        ),
        energy: 0.07,
        broadening: 0.12,
    };
    let tangent = SurfaceGreenTangent {
        cell_hamiltonian: matrix(
            2,
            2,
            &[(0.2, 0.0), (-0.03, 0.05), (-0.03, -0.05), (-0.1, 0.0)],
        ),
        inter_cell_hopping: matrix(
            2,
            2,
            &[(0.06, -0.02), (0.03, 0.04), (-0.05, 0.01), (-0.02, 0.03)],
        ),
        energy: -0.08,
        broadening: 0.04,
    };
    let output_cotangent = matrix(
        2,
        2,
        &[(0.2, -0.1), (-0.03, 0.08), (0.05, -0.04), (-0.15, 0.06)],
    );
    let rule = SurfaceGreenRule::new(1.0e-14, 512).unwrap();
    let (_, green_tangent) = rule.jvp(&arguments, &tangent).unwrap();
    let (_, pullback) = rule.vjp(&arguments).unwrap();
    let input_cotangent = pullback.apply(output_cotangent.clone()).unwrap();

    let output_pairing = real_frobenius_pairing(&green_tangent, &output_cotangent).unwrap();
    let input_pairing =
        real_frobenius_pairing(&tangent.cell_hamiltonian, &input_cotangent.cell_hamiltonian)
            .unwrap()
            + real_frobenius_pairing(
                &tangent.inter_cell_hopping,
                &input_cotangent.inter_cell_hopping,
            )
            .unwrap()
            + tangent.energy * input_cotangent.energy
            + tangent.broadening * input_cotangent.broadening;
    assert_close(output_pairing, input_pairing);

    let displaced = |scale: f64| SurfaceGreenArguments {
        cell_hamiltonian: ComplexMatrix::new(
            2,
            2,
            arguments
                .cell_hamiltonian
                .as_slice()
                .iter()
                .zip(tangent.cell_hamiltonian.as_slice())
                .map(|(value, direction)| value + scale * direction)
                .collect(),
        )
        .unwrap(),
        inter_cell_hopping: ComplexMatrix::new(
            2,
            2,
            arguments
                .inter_cell_hopping
                .as_slice()
                .iter()
                .zip(tangent.inter_cell_hopping.as_slice())
                .map(|(value, direction)| value + scale * direction)
                .collect(),
        )
        .unwrap(),
        energy: arguments.energy + scale * tangent.energy,
        broadening: arguments.broadening + scale * tangent.broadening,
    };
    let positive = rule.value(&displaced(FD_STEP)).unwrap();
    let negative = rule.value(&displaced(-FD_STEP)).unwrap();
    for ((positive, negative), analytic) in positive
        .as_slice()
        .iter()
        .zip(negative.as_slice())
        .zip(green_tangent.as_slice())
    {
        let numerical = (positive - negative) / (2.0 * FD_STEP);
        assert_close(numerical.re, analytic.re);
        assert_close(numerical.im, analytic.im);
    }
}

#[test]
fn quantum_metric_mesh_gradient_is_gauge_invariant_and_matches_finite_difference() {
    let point_count = 7;
    let momentum_step = std::f64::consts::TAU / point_count as f64;
    let families = (0..point_count)
        .map(|point| {
            let momentum = point as f64 * momentum_step;
            let base = matrix(
                2,
                2,
                &[
                    (0.7, 0.0),
                    (momentum.cos(), -momentum.sin()),
                    (momentum.cos(), momentum.sin()),
                    (-0.7, 0.0),
                ],
            );
            let mass = matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)]);
            let mixing = matrix(2, 2, &[(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]);
            AffineHermitianFamily::new(base, vec![mass, mixing]).unwrap()
        })
        .collect::<Vec<_>>();
    let swapped_families = families
        .iter()
        .map(|family| {
            let swap = |source: &ComplexMatrix| {
                matrix(
                    2,
                    2,
                    &[
                        (source.as_slice()[3].re, source.as_slice()[3].im),
                        (source.as_slice()[2].re, source.as_slice()[2].im),
                        (source.as_slice()[1].re, source.as_slice()[1].im),
                        (source.as_slice()[0].re, source.as_slice()[0].im),
                    ],
                )
            };
            AffineHermitianFamily::new(
                swap(family.base()),
                family.directions().iter().map(swap).collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let objective = QuantumMetricMeshObjective::new(families, 1, momentum_step, 1.0e-4).unwrap();
    let swapped =
        QuantumMetricMeshObjective::new(swapped_families, 1, momentum_step, 1.0e-4).unwrap();
    let parameters = ModelParameters::new(vec![0.12, -0.08]).unwrap();
    let direction = ModelDirection::new(vec![0.3, -0.4]).unwrap();

    let (value, directional) = objective.jvp(&parameters, &direction).unwrap();
    assert_close(value, swapped.value(&parameters).unwrap());
    let positive = objective
        .value(&parameters.displaced(&direction, FD_STEP).unwrap())
        .unwrap();
    let negative = objective
        .value(&parameters.displaced(&direction, -FD_STEP).unwrap())
        .unwrap();
    assert_close((positive - negative) / (2.0 * FD_STEP), directional);
    let (_, gradient) = objective.value_and_grad(&parameters).unwrap();
    let reverse_directional = gradient
        .as_slice()
        .iter()
        .zip(direction.as_slice())
        .map(|(gradient, direction)| gradient * direction)
        .sum::<f64>();
    assert_close(reverse_directional, directional);
}

#[test]
fn complete_device_and_lead_transmission_vjp_matches_finite_difference() {
    let system = DifferentiableOpenSystem {
        device_hamiltonian: matrix(1, 1, &[(0.1, 0.0)]),
        leads: vec![
            DifferentiableLead {
                cell_hamiltonian: matrix(1, 1, &[(-0.04, 0.0)]),
                inter_cell_hopping: matrix(1, 1, &[(-0.82, 0.03)]),
                coupling: matrix(1, 1, &[(-0.51, 0.02)]),
                broadening: 0.08,
            },
            DifferentiableLead {
                cell_hamiltonian: matrix(1, 1, &[(0.06, 0.0)]),
                inter_cell_hopping: matrix(1, 1, &[(-0.73, -0.02)]),
                coupling: matrix(1, 1, &[(-0.43, -0.04)]),
                broadening: 0.09,
            },
        ],
        energy: 0.12,
    };
    let direction = OpenSystemDirection {
        device_hamiltonian: matrix(1, 1, &[(0.07, 0.0)]),
        leads: vec![
            LeadDirection {
                cell_hamiltonian: matrix(1, 1, &[(0.03, 0.0)]),
                inter_cell_hopping: matrix(1, 1, &[(0.02, -0.01)]),
                coupling: matrix(1, 1, &[(-0.015, 0.012)]),
                broadening: 0.01,
            },
            LeadDirection {
                cell_hamiltonian: matrix(1, 1, &[(-0.025, 0.0)]),
                inter_cell_hopping: matrix(1, 1, &[(-0.018, 0.009)]),
                coupling: matrix(1, 1, &[(0.011, -0.007)]),
                broadening: -0.008,
            },
        ],
        energy: -0.04,
    };
    let objective = OpenSystemTransmission::new(1, 0, 1.0e-14, 512).unwrap();
    let (value, analytic) = objective.jvp(&system, &direction).unwrap();
    assert!(value > 0.0);

    let shifted_matrix = |primal: &ComplexMatrix, tangent: &ComplexMatrix, scale: f64| {
        ComplexMatrix::new(
            primal.rows(),
            primal.columns(),
            primal
                .as_slice()
                .iter()
                .zip(tangent.as_slice())
                .map(|(primal, tangent)| primal + scale * tangent)
                .collect(),
        )
        .unwrap()
    };
    let displaced = |scale: f64| DifferentiableOpenSystem {
        device_hamiltonian: shifted_matrix(
            &system.device_hamiltonian,
            &direction.device_hamiltonian,
            scale,
        ),
        leads: system
            .leads
            .iter()
            .zip(&direction.leads)
            .map(|(lead, tangent)| DifferentiableLead {
                cell_hamiltonian: shifted_matrix(
                    &lead.cell_hamiltonian,
                    &tangent.cell_hamiltonian,
                    scale,
                ),
                inter_cell_hopping: shifted_matrix(
                    &lead.inter_cell_hopping,
                    &tangent.inter_cell_hopping,
                    scale,
                ),
                coupling: shifted_matrix(&lead.coupling, &tangent.coupling, scale),
                broadening: lead.broadening + scale * tangent.broadening,
            })
            .collect(),
        energy: system.energy + scale * direction.energy,
    };
    let positive = objective.value(&displaced(FD_STEP)).unwrap();
    let negative = objective.value(&displaced(-FD_STEP)).unwrap();
    assert_close((positive - negative) / (2.0 * FD_STEP), analytic);

    let (_, pullback) = objective.vjp(&system).unwrap();
    let half_gradient = pullback.apply(0.5).unwrap();
    let (_, gradient) = objective.value_and_grad(&system).unwrap();
    assert_close(
        real_frobenius_pairing(
            &half_gradient.device_hamiltonian,
            &direction.device_hamiltonian,
        )
        .unwrap(),
        0.5 * real_frobenius_pairing(&gradient.device_hamiltonian, &direction.device_hamiltonian)
            .unwrap(),
    );
}

#[test]
fn nonsmooth_primal_and_pullback_failures_are_structured() {
    let degenerate = ComplexMatrix::zeros(2, 2);
    let direction = ComplexMatrix::identity(2);
    let eigenvalue = IsolatedEigenvalue::new(0, 1.0e-5).unwrap();
    assert!(matches!(
        eigenvalue.jvp(&degenerate, &direction),
        Err(AdError::GapTooSmall { .. })
    ));

    let singular = LinearSolveArguments {
        matrix: ComplexMatrix::zeros(2, 2),
        right_hand_side: vec![Complex64::new(1.0, 0.0); 2],
    };
    assert!(matches!(
        LinearSolve.vjp(&singular),
        Err(AdError::SingularPrimal)
    ));

    let regular = LinearSolveArguments {
        matrix: ComplexMatrix::identity(2),
        right_hand_side: vec![Complex64::new(1.0, 0.0); 2],
    };
    let (_, pullback) = LinearSolve.vjp(&regular).unwrap();
    assert!(matches!(
        pullback.apply(vec![Complex64::new(1.0, 0.0)]),
        Err(AdError::Shape { .. })
    ));

    let invalid_surface = SurfaceGreenArguments {
        cell_hamiltonian: ComplexMatrix::identity(1),
        inter_cell_hopping: ComplexMatrix::scalar(Complex64::new(0.1, 0.0)),
        energy: 0.0,
        broadening: 0.0,
    };
    assert!(matches!(
        SurfaceGreenRule::default().value(&invalid_surface),
        Err(AdError::Transport(_))
    ));
}
