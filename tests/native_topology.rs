use thouless::model::{Lattice, ModelBuilder};
use thouless::topology::{
    chern_numbers_on_uniform_grid, connection_from_link, local_chern_marker_from_hamiltonian,
    local_chern_marker_from_projector, parallel_transport_link, plaquette_flux,
    quantum_geometric_tensor_from_hamiltonian_derivatives, reduced_polarization_on_loop,
    second_chern_from_hamiltonian_derivatives, unitary_matrix_power, wilson_line_phase,
    wilson_loop_eigenphases, z2_from_wannier_centers, z2_invariant_on_uniform_grid,
};
use thouless::{Complex64, ComplexMatrix};

fn frame(values: &[Complex64]) -> ComplexMatrix {
    ComplexMatrix::new(1, values.len(), values.to_vec()).unwrap()
}

#[test]
fn fractional_unitary_power_reconstructs_the_holonomy() {
    let phase = Complex64::from_polar(1.0, 0.73);
    let link = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            phase,
            phase,
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();

    let root = unitary_matrix_power(&link, 0.25).unwrap();
    let root = nalgebra::DMatrix::from_row_slice(2, 2, root.as_slice());
    let reconstructed = &root * &root * &root * &root;

    for (actual, expected) in reconstructed.iter().zip(link.as_slice()) {
        assert!((*actual - expected).norm() < 1.0e-10);
    }
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

#[test]
fn massive_dirac_quantum_geometry_matches_the_kubo_tensor() {
    let mass = 2.0;
    let hamiltonian = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(mass, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-mass, 0.0),
        ],
    )
    .unwrap();
    let sigma_x = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_y = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let tensor = quantum_geometric_tensor_from_hamiltonian_derivatives(
        &hamiltonian,
        &[sigma_x, sigma_y],
        &[0],
    )
    .unwrap();
    let diagonal = 1.0 / (4.0 * mass * mass);
    assert_eq!(tensor.direction_count(), 2);
    assert_eq!(tensor.occupied_count(), 1);
    assert!(
        (tensor.component(0, 0).unwrap().get(0, 0).unwrap() - Complex64::new(diagonal, 0.0)).norm()
            < 1.0e-12
    );
    assert!(
        (tensor.component(1, 1).unwrap().get(0, 0).unwrap() - Complex64::new(diagonal, 0.0)).norm()
            < 1.0e-12
    );
    assert!(
        (tensor.component(0, 1).unwrap().get(0, 0).unwrap() - Complex64::new(0.0, -diagonal))
            .norm()
            < 1.0e-12
    );
}

#[test]
fn local_chern_marker_uses_the_occupied_projector_not_an_eigenvector_gauge() {
    let dimension = 4;
    let uniform = vec![Complex64::new(0.5, 0.0); dimension];
    let mut second = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 2.0),
        Complex64::new(-1.0, 1.0),
        Complex64::new(0.3, -0.7),
    ];
    let overlap = uniform
        .iter()
        .zip(&second)
        .map(|(left, right)| left.conj() * right)
        .sum::<Complex64>();
    for (value, uniform_value) in second.iter_mut().zip(&uniform) {
        *value -= uniform_value * overlap;
    }
    let norm = second.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    for value in &mut second {
        *value /= norm;
    }

    let projector_values = (0..dimension)
        .flat_map(|row| {
            let uniform = &uniform;
            let second = &second;
            (0..dimension).map(move |column| {
                uniform[row] * uniform[column].conj() + second[row] * second[column].conj()
            })
        })
        .collect::<Vec<_>>();
    let projector = ComplexMatrix::new(dimension, dimension, projector_values).unwrap();
    let hamiltonian_values = (0..dimension)
        .flat_map(|row| {
            let projector = &projector;
            (0..dimension).map(move |column| {
                let identity = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                identity - 2.0 * projector.get(row, column).unwrap()
            })
        })
        .collect();
    let hamiltonian = ComplexMatrix::new(dimension, dimension, hamiltonian_values).unwrap();
    let positions = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let direct = local_chern_marker_from_projector(&projector, &positions, 1.0).unwrap();
    let spectral =
        local_chern_marker_from_hamiltonian(&hamiltonian, &positions, &[0, 1], 1.0).unwrap();
    for (left, right) in direct.iter().zip(&spectral) {
        assert!((left - right).abs() < 1.0e-10);
    }
    assert!(direct.iter().any(|marker| marker.abs() > 1.0e-3));
    assert!(direct.iter().sum::<f64>().abs() < 1.0e-12);
}

#[test]
fn uniform_grid_chern_number_recovers_two_band_topology() {
    let lattice = Lattice::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0, 1]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder
        .add_orbital_with_dof("spinor", [0.0, 0.0], 2)
        .unwrap();
    let sigma_x = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_y = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_z = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap();
    builder
        .set_onsite_block(orbital, scaled(&sigma_z, -1.0))
        .unwrap();
    builder
        .add_hopping_block(
            orbital,
            orbital,
            [1, 0],
            add_scaled(&sigma_z, 0.5, &sigma_x, Complex64::new(0.0, -0.5)),
        )
        .unwrap();
    builder
        .add_hopping_block(
            orbital,
            orbital,
            [0, 1],
            add_scaled(&sigma_z, 0.5, &sigma_y, Complex64::new(0.0, -0.5)),
        )
        .unwrap();
    let model = builder.build().unwrap();

    let result = chern_numbers_on_uniform_grid(&model, &[31, 31], [0, 1], &[0]).unwrap();
    assert!(result.spectator_shape().is_empty());
    assert_eq!(result.values().len(), 1);
    assert!((result.values()[0].abs() - 1.0).abs() < 1.0e-10);
}

#[test]
fn kubo_second_chern_recovers_four_dimensional_dirac_topology() {
    let sigma_x = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_y = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_z = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap();
    let identity = ComplexMatrix::identity(2);
    let gammas = [
        kronecker(&sigma_x, &sigma_x),
        kronecker(&sigma_x, &sigma_y),
        kronecker(&sigma_x, &sigma_z),
        kronecker(&sigma_y, &identity),
        kronecker(&sigma_z, &identity),
    ];

    let samples = 9_usize;
    let step = 1.0 / samples as f64;
    let mass = -3.0;
    let grid_size = samples.pow(4);
    let mut hamiltonians = Vec::with_capacity(grid_size);
    let mut derivatives = Vec::with_capacity(grid_size);
    for flat_index in 0..grid_size {
        let mut remainder = flat_index;
        let mut coordinates = [0.0; 4];
        for axis in (0..4).rev() {
            coordinates[axis] = (remainder % samples) as f64 * step;
            remainder /= samples;
        }
        let angles = coordinates.map(|coordinate| 2.0 * std::f64::consts::PI * coordinate);
        let mut coefficients = [0.0; 5];
        for axis in 0..4 {
            coefficients[axis] = angles[axis].sin();
        }
        coefficients[4] = mass + angles.iter().map(|angle| angle.cos()).sum::<f64>();
        hamiltonians.push(linear_combination(&gammas, &coefficients));
        derivatives.push(std::array::from_fn(|axis| {
            linear_combination(
                &gammas,
                &[
                    if axis == 0 {
                        2.0 * std::f64::consts::PI * angles[axis].cos()
                    } else {
                        0.0
                    },
                    if axis == 1 {
                        2.0 * std::f64::consts::PI * angles[axis].cos()
                    } else {
                        0.0
                    },
                    if axis == 2 {
                        2.0 * std::f64::consts::PI * angles[axis].cos()
                    } else {
                        0.0
                    },
                    if axis == 3 {
                        2.0 * std::f64::consts::PI * angles[axis].cos()
                    } else {
                        0.0
                    },
                    -2.0 * std::f64::consts::PI * angles[axis].sin(),
                ],
            )
        }));
    }

    let result = second_chern_from_hamiltonian_derivatives(
        &hamiltonians,
        &derivatives,
        &[samples; 4],
        &[step; 4],
        true,
        &[0, 1],
    )
    .unwrap();
    assert_eq!(result.slice_densities().len(), samples);
    assert!(
        (result.value().abs() - 1.0).abs() < 0.08,
        "expected unit second Chern magnitude, got {}",
        result.value()
    );
}

fn block_diagonal(first: &ComplexMatrix, second: &ComplexMatrix) -> ComplexMatrix {
    let dimension = first.rows();
    let mut result = ComplexMatrix::zeros(2 * dimension, 2 * dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            result
                .set(row, column, first.get(row, column).unwrap())
                .unwrap();
            result
                .set(
                    dimension + row,
                    dimension + column,
                    second.get(row, column).unwrap(),
                )
                .unwrap();
        }
    }
    result
}

fn quantum_spin_hall_model(mass: f64) -> thouless::model::TightBindingModel {
    let lattice = Lattice::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0, 1]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder
        .add_orbital_with_dof("spinful-orbital", [0.0, 0.0], 4)
        .unwrap();
    let sigma_x = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_y = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_z = ComplexMatrix::new(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap();
    let onsite = scaled(&sigma_z, mass);
    let up_x = add_scaled(&sigma_z, 0.5, &sigma_x, Complex64::new(0.0, -0.5));
    let down_x = add_scaled(&sigma_z, 0.5, &sigma_x, Complex64::new(0.0, 0.5));
    let y_hopping = add_scaled(&sigma_z, 0.5, &sigma_y, Complex64::new(0.0, -0.5));
    builder
        .set_onsite_block(orbital, block_diagonal(&onsite, &onsite))
        .unwrap();
    builder
        .add_hopping_block(orbital, orbital, [1, 0], block_diagonal(&up_x, &down_x))
        .unwrap();
    builder
        .add_hopping_block(
            orbital,
            orbital,
            [0, 1],
            block_diagonal(&y_hopping, &y_hopping),
        )
        .unwrap();
    builder.build().unwrap()
}

#[test]
fn atomic_wannier_center_is_the_reduced_polarization() {
    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).unwrap();
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder.add_orbital("atomic", [0.3]).unwrap();
    builder.set_onsite(orbital, -1.0).unwrap();
    let model = builder.build().unwrap();

    let polarization = reduced_polarization_on_loop(&model, 8, 0, &[0.0], &[0]).unwrap();
    assert!((polarization - 0.3).abs() < 1.0e-12);
}

#[test]
fn largest_gap_flow_distinguishes_even_and_odd_partner_switching() {
    let odd = z2_from_wannier_centers(
        &[
            vec![0.0, 0.0],
            vec![0.2, 0.8],
            vec![0.4, 0.6],
            vec![0.5, 0.5],
        ],
        1.0e-10,
    )
    .unwrap();
    let even = z2_from_wannier_centers(&[vec![0.0, 0.0], vec![0.1, 0.9], vec![0.0, 0.0]], 1.0e-10)
        .unwrap();
    assert_eq!(odd.value(), 1);
    assert_eq!(odd.crossing_count() % 2, 1);
    assert_eq!(even.value(), 0);
}

#[test]
fn quantum_spin_hall_z2_is_stable_under_mesh_refinement() {
    let topological = quantum_spin_hall_model(-1.0);
    let coarse =
        z2_invariant_on_uniform_grid(&topological, 31, 21, [0, 1], &[0, 1], 1.0e-7).unwrap();
    let refined =
        z2_invariant_on_uniform_grid(&topological, 51, 31, [0, 1], &[0, 1], 1.0e-7).unwrap();
    let trivial = z2_invariant_on_uniform_grid(
        &quantum_spin_hall_model(-3.0),
        31,
        21,
        [0, 1],
        &[0, 1],
        1.0e-7,
    )
    .unwrap();

    assert_eq!(coarse.value(), 1);
    assert_eq!(refined.value(), 1);
    assert_eq!(trivial.value(), 0);
}

fn scaled(matrix: &ComplexMatrix, factor: f64) -> ComplexMatrix {
    ComplexMatrix::new(
        matrix.rows(),
        matrix.columns(),
        matrix
            .as_slice()
            .iter()
            .map(|value| *value * factor)
            .collect(),
    )
    .unwrap()
}

fn kronecker(left: &ComplexMatrix, right: &ComplexMatrix) -> ComplexMatrix {
    let rows = left.rows() * right.rows();
    let columns = left.columns() * right.columns();
    let mut result = ComplexMatrix::zeros(rows, columns);
    for left_row in 0..left.rows() {
        for left_column in 0..left.columns() {
            for right_row in 0..right.rows() {
                for right_column in 0..right.columns() {
                    result
                        .set(
                            left_row * right.rows() + right_row,
                            left_column * right.columns() + right_column,
                            left.get(left_row, left_column).unwrap()
                                * right.get(right_row, right_column).unwrap(),
                        )
                        .unwrap();
                }
            }
        }
    }
    result
}

fn linear_combination(matrices: &[ComplexMatrix; 5], coefficients: &[f64; 5]) -> ComplexMatrix {
    ComplexMatrix::new(
        matrices[0].rows(),
        matrices[0].columns(),
        (0..matrices[0].rows() * matrices[0].columns())
            .map(|index| {
                matrices
                    .iter()
                    .zip(coefficients)
                    .map(|(matrix, coefficient)| matrix.as_slice()[index] * coefficient)
                    .sum()
            })
            .collect(),
    )
    .unwrap()
}

fn add_scaled(
    left: &ComplexMatrix,
    left_factor: f64,
    right: &ComplexMatrix,
    right_factor: Complex64,
) -> ComplexMatrix {
    ComplexMatrix::new(
        left.rows(),
        left.columns(),
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(left, right)| *left * left_factor + *right * right_factor)
            .collect(),
    )
    .unwrap()
}
