use thouless::bands::PeriodicBands;
use thouless::continuum::{finite_difference_stencil, DifferentialFactor};
use thouless::decomposition::schur;
use thouless::digest::{gaussian as digest_gaussian, uniform};
use thouless::geometry::ReciprocalPath;
use thouless::graph::{CompressionOptions, DirectedEdge, DirectedGraphBuilder};
use thouless::interpolation::{interpolate_density, SmoothingOptions};
use thouless::kpm::rescale_hamiltonian;
use thouless::lattice_reduction::lll_reduce;
use thouless::lead_modes::propagating_modes;
use thouless::linear_operator::CsrMatrix;
use thouless::model::Lattice;
use thouless::observables::project_diagonal_observable;
use thouless::periodic::bloch_phase;
use thouless::random_matrix::{gaussian_from_components, SymmetryClass};
use thouless::response::{
    intrinsic_berry_curvature_from_model, FermiDistribution, MomentumCoordinates,
};
use thouless::sparse_direct::SparseLuFactorization;
use thouless::spectrum::hermitian_eigensystem;
use thouless::symmetry::{particle_hole_symmetric_basis, DiscreteSymmetry};
use thouless::topology::{
    chern_numbers_on_uniform_grid, local_chern_marker_from_hamiltonian,
    quantum_geometric_tensor_from_hamiltonian_derivatives, wilson_line_phase,
};
use thouless::transport::{
    regularize_retarded_self_energy, retarded_lead_self_energy, solve_open_system, LeadContact,
    SurfaceGreenOptions,
};
use thouless::wannier::project_trials;
use thouless::{Complex64, RealMatrix};

use crate::model::model_ref;
use crate::{
    borrowed_mut_slice, borrowed_slice, boundary, read_complex_matrix, read_complex_tensor3,
    read_real_matrix, write_complex_matrix, write_complex_tensor3, write_real_matrix, AbiError,
    ThoulessC64MatrixMut, ThoulessC64MatrixView, ThoulessC64Tensor3Mut, ThoulessC64Tensor3View,
    ThoulessComplex64, ThoulessF64MatrixMut, ThoulessF64MatrixView, ThoulessLeadView,
    ThoulessModel, ThoulessStatus,
};

fn write_f64(
    values: &[f64],
    output: *mut f64,
    capacity: usize,
    name: &str,
) -> Result<(), AbiError> {
    if capacity < values.len() {
        return Err(AbiError::new(
            ThoulessStatus::BufferTooSmall,
            format!("{name} has {capacity} elements; required {}", values.len()),
        ));
    }
    let destination = unsafe { borrowed_mut_slice(output, capacity, name)? };
    destination[..values.len()].copy_from_slice(values);
    Ok(())
}

fn write_usize(
    values: &[usize],
    output: *mut usize,
    capacity: usize,
    name: &str,
) -> Result<(), AbiError> {
    if capacity < values.len() {
        return Err(AbiError::new(
            ThoulessStatus::BufferTooSmall,
            format!("{name} has {capacity} elements; required {}", values.len()),
        ));
    }
    let destination = unsafe { borrowed_mut_slice(output, capacity, name)? };
    destination[..values.len()].copy_from_slice(values);
    Ok(())
}

fn real_rows(matrix: &RealMatrix) -> Vec<Vec<f64>> {
    (0..matrix.rows())
        .map(|row| matrix.as_slice()[row * matrix.columns()..(row + 1) * matrix.columns()].to_vec())
        .collect()
}

fn symmetry_class(value: u32) -> Result<SymmetryClass, AbiError> {
    match value {
        0 => Ok(SymmetryClass::A),
        1 => Ok(SymmetryClass::Ai),
        2 => Ok(SymmetryClass::Aii),
        3 => Ok(SymmetryClass::Aiii),
        4 => Ok(SymmetryClass::Bdi),
        5 => Ok(SymmetryClass::Cii),
        6 => Ok(SymmetryClass::D),
        7 => Ok(SymmetryClass::Diii),
        8 => Ok(SymmetryClass::C),
        9 => Ok(SymmetryClass::Ci),
        _ => Err(AbiError::invalid("unknown Altland-Zirnbauer class")),
    }
}

/// Diagonalize a dense Hermitian matrix.
#[no_mangle]
pub unsafe extern "C" fn thouless_hermitian_eigensystem(
    matrix: ThoulessC64MatrixView,
    eigenvalues: *mut f64,
    eigenvalue_capacity: usize,
    eigenvectors: ThoulessC64MatrixMut,
) -> ThoulessStatus {
    boundary(|| {
        let matrix = unsafe { read_complex_matrix(matrix, "Hermitian matrix")? };
        let solution = hermitian_eigensystem(&matrix, 1.0e-12)
            .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        write_f64(
            solution.eigenvalues(),
            eigenvalues,
            eigenvalue_capacity,
            "eigenvalues",
        )?;
        unsafe { write_complex_matrix(solution.eigenvectors(), eigenvectors, "eigenvectors") }
    })
}

/// Rescale a dense Hamiltonian strictly inside the Chebyshev interval.
#[no_mangle]
pub unsafe extern "C" fn thouless_kpm_rescale_dense(
    hamiltonian: ThoulessC64MatrixView,
    strict_margin: f64,
    use_explicit_bounds: bool,
    lower_bound: f64,
    upper_bound: f64,
    output: ThoulessC64MatrixMut,
    half_width: *mut f64,
    center: *mut f64,
) -> ThoulessStatus {
    boundary(|| {
        if half_width.is_null() || center.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "KPM scale output is null",
            ));
        }
        let hamiltonian = unsafe { read_complex_matrix(hamiltonian, "Hamiltonian")? };
        let rescaled = rescale_hamiltonian(
            &hamiltonian,
            strict_margin,
            use_explicit_bounds.then_some((lower_bound, upper_bound)),
        )
        .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe {
            *half_width = rescaled.scale().half_width();
            *center = rescaled.scale().center();
            write_complex_matrix(rescaled.matrix(), output, "rescaled Hamiltonian")
        }
    })
}

/// Evaluate periodic lead energies and derivatives through order two.
#[no_mangle]
pub unsafe extern "C" fn thouless_lead_bands(
    cell_hamiltonian: ThoulessC64MatrixView,
    inter_cell_hopping: ThoulessC64MatrixView,
    momentum: f64,
    derivative_order: usize,
    energies: *mut f64,
    energy_capacity: usize,
    first_derivatives: *mut f64,
    first_capacity: usize,
    second_derivatives: *mut f64,
    second_capacity: usize,
) -> ThoulessStatus {
    boundary(|| {
        let bands = PeriodicBands::new(
            unsafe { read_complex_matrix(cell_hamiltonian, "lead cell Hamiltonian")? },
            unsafe { read_complex_matrix(inter_cell_hopping, "lead hopping")? },
        )
        .map_err(|error| AbiError::invalid(error.to_string()))?;
        let result = bands
            .evaluate(momentum, derivative_order, false)
            .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        write_f64(
            result.energies(),
            energies,
            energy_capacity,
            "band energies",
        )?;
        if let Some(values) = result.first_derivatives() {
            write_f64(
                values,
                first_derivatives,
                first_capacity,
                "first derivatives",
            )?;
        }
        if let Some(values) = result.second_derivatives() {
            write_f64(
                values,
                second_derivatives,
                second_capacity,
                "second derivatives",
            )?;
        }
        Ok(())
    })
}

/// Evaluate an integer-translation Bloch phase.
#[no_mangle]
pub unsafe extern "C" fn thouless_bloch_phase(
    translation: *const i64,
    translation_length: usize,
    momentum: *const f64,
    momentum_length: usize,
    output: *mut ThoulessComplex64,
) -> ThoulessStatus {
    boundary(|| {
        if output.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "Bloch phase output is null",
            ));
        }
        let translation =
            unsafe { borrowed_slice(translation, translation_length, "translation")? };
        let momentum = unsafe { borrowed_slice(momentum, momentum_length, "momentum")? };
        let phase = bloch_phase(translation, momentum)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe { *output = phase.into() };
        Ok(())
    })
}

/// Sample a metric-aware reciprocal path.
#[no_mangle]
pub unsafe extern "C" fn thouless_reciprocal_path(
    primitive_vectors: ThoulessF64MatrixView,
    periodic_axes: *const usize,
    periodic_axis_count: usize,
    nodes: ThoulessF64MatrixView,
    sample_count: usize,
    points: ThoulessF64MatrixMut,
    distances: *mut f64,
    distance_capacity: usize,
    node_distances: *mut f64,
    node_distance_capacity: usize,
) -> ThoulessStatus {
    boundary(|| {
        let primitive = unsafe { read_real_matrix(primitive_vectors, "primitive vectors")? };
        let axes = unsafe { borrowed_slice(periodic_axes, periodic_axis_count, "periodic axes")? };
        let nodes = unsafe { read_real_matrix(nodes, "path nodes")? };
        let lattice = Lattice::new(real_rows(&primitive), axes.to_vec())
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        let path = ReciprocalPath::through(&lattice, &real_rows(&nodes), sample_count)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        let point_matrix = RealMatrix::new(
            path.reduced_points().len(),
            periodic_axis_count,
            path.reduced_points().iter().flatten().copied().collect(),
        )
        .map_err(|error| AbiError::shape(error.to_string()))?;
        unsafe { write_real_matrix(&point_matrix, points, "path points")? };
        write_f64(
            path.distances(),
            distances,
            distance_capacity,
            "path distances",
        )?;
        write_f64(
            path.node_distances(),
            node_distances,
            node_distance_capacity,
            "node distances",
        )
    })
}

/// Construct a centered finite-difference stencil for momentum powers.
///
/// `axes` and `powers` contain one entry per ordered momentum factor. Output
/// offsets and inverse-spacing powers use row-major `(term, dimension)`
/// storage. Call once with zero output capacity to query `term_count`.
#[no_mangle]
pub unsafe extern "C" fn thouless_continuum_momentum_stencil(
    dimension: usize,
    axes: *const usize,
    powers: *const usize,
    factor_count: usize,
    offsets: *mut i32,
    inverse_spacing_powers: *mut u32,
    weights: *mut ThoulessComplex64,
    term_capacity: usize,
    term_count: *mut usize,
) -> ThoulessStatus {
    boundary(|| {
        if term_count.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "stencil term count is null",
            ));
        }
        let axes = unsafe { borrowed_slice(axes, factor_count, "momentum axes")? };
        let powers = unsafe { borrowed_slice(powers, factor_count, "momentum powers")? };
        let factors = axes
            .iter()
            .copied()
            .zip(powers.iter().copied())
            .map(|(axis, power)| DifferentialFactor::Momentum { axis, power })
            .collect::<Vec<_>>();
        let terms = finite_difference_stencil(dimension, &factors)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe { *term_count = terms.len() };
        if term_capacity < terms.len() {
            return Err(AbiError::new(
                ThoulessStatus::BufferTooSmall,
                format!(
                    "stencil output has {term_capacity} terms; required {}",
                    terms.len()
                ),
            ));
        }
        let element_count = terms.len().checked_mul(dimension).ok_or_else(|| {
            AbiError::new(
                ThoulessStatus::ResourceExhausted,
                "stencil output size overflow",
            )
        })?;
        let offsets = unsafe { borrowed_mut_slice(offsets, element_count, "stencil offsets")? };
        let inverse = unsafe {
            borrowed_mut_slice(
                inverse_spacing_powers,
                element_count,
                "stencil inverse-spacing powers",
            )?
        };
        let weights = unsafe { borrowed_mut_slice(weights, terms.len(), "stencil weights")? };
        for (term_index, term) in terms.iter().enumerate() {
            weights[term_index] = term.weight().into();
            let start = term_index * dimension;
            offsets[start..start + dimension].copy_from_slice(term.wave_offset());
            inverse[start..start + dimension].copy_from_slice(term.inverse_spacing_powers());
        }
        Ok(())
    })
}

/// Interpolate discrete density values onto a regular Cartesian field.
///
/// Reference-edge endpoint rows determine the default smoothing scale. The
/// function writes shape and bounds on both successful and size-query calls.
#[no_mangle]
pub unsafe extern "C" fn thouless_interpolate_density(
    points: ThoulessF64MatrixView,
    values: *const f64,
    value_count: usize,
    reference_starts: ThoulessF64MatrixView,
    reference_ends: ThoulessF64MatrixView,
    absolute_width: f64,
    samples_per_width: usize,
    shape: *mut usize,
    shape_capacity: usize,
    bounds: ThoulessF64MatrixMut,
    output_values: *mut f64,
    output_capacity: usize,
    required_values: *mut usize,
) -> ThoulessStatus {
    boundary(|| {
        if required_values.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "field value count is null",
            ));
        }
        let points = unsafe { read_real_matrix(points, "density points")? };
        let values = unsafe { borrowed_slice(values, value_count, "density values")? };
        let starts = unsafe { read_real_matrix(reference_starts, "reference starts")? };
        let ends = unsafe { read_real_matrix(reference_ends, "reference ends")? };
        if starts.shape() != ends.shape() || starts.columns() != points.columns() {
            return Err(AbiError::shape(
                "reference edge endpoints must share the point dimension",
            ));
        }
        let point_rows = real_rows(&points);
        let edge_starts = real_rows(&starts);
        let edge_ends = real_rows(&ends);
        let edges = edge_starts.into_iter().zip(edge_ends).collect::<Vec<_>>();
        let field = interpolate_density(
            &point_rows,
            values,
            &edges,
            SmoothingOptions {
                absolute_width: Some(absolute_width),
                relative_width: None,
                samples_per_width,
            },
        )
        .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe { *required_values = field.values().len() };
        write_usize(field.shape(), shape, shape_capacity, "field shape")?;
        let bounds_matrix = RealMatrix::new(
            field.bounds().len(),
            2,
            field
                .bounds()
                .iter()
                .flat_map(|(lower, upper)| [*lower, *upper])
                .collect(),
        )
        .map_err(|error| AbiError::shape(error.to_string()))?;
        unsafe { write_real_matrix(&bounds_matrix, bounds, "field bounds")? };
        write_f64(
            field.values(),
            output_values,
            output_capacity,
            "field values",
        )
    })
}

/// Compute the scalar Wilson-loop phase of sampled orthonormal frames.
#[no_mangle]
pub unsafe extern "C" fn thouless_wilson_phase(
    frames: ThoulessC64Tensor3View,
    output: *mut f64,
) -> ThoulessStatus {
    boundary(|| {
        if output.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "Wilson phase output is null",
            ));
        }
        let frames = unsafe { read_complex_tensor3(frames, "Wilson frames")? };
        let phase =
            wilson_line_phase(&frames).map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe { *output = phase };
        Ok(())
    })
}

/// Compute occupied-subspace Chern numbers on a uniform model grid.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_chern_numbers(
    model: *const ThoulessModel,
    samples: *const usize,
    sample_count: usize,
    first_direction: usize,
    second_direction: usize,
    occupied_states: *const usize,
    occupied_count: usize,
    output: *mut f64,
    output_capacity: usize,
    value_count: *mut usize,
) -> ThoulessStatus {
    boundary(|| {
        if value_count.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "Chern value count is null",
            ));
        }
        let model = unsafe { model_ref(model)? };
        let samples = unsafe { borrowed_slice(samples, sample_count, "Chern samples")? };
        let occupied =
            unsafe { borrowed_slice(occupied_states, occupied_count, "occupied states")? };
        let result = chern_numbers_on_uniform_grid(
            &model.inner,
            samples,
            [first_direction, second_direction],
            occupied,
        )
        .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        unsafe { *value_count = result.values().len() };
        write_f64(result.values(), output, output_capacity, "Chern numbers")
    })
}

/// Compute a non-Abelian Kubo quantum-geometric tensor.
#[no_mangle]
pub unsafe extern "C" fn thouless_quantum_geometric_tensor(
    hamiltonian: ThoulessC64MatrixView,
    derivatives: ThoulessC64Tensor3View,
    occupied_states: *const usize,
    occupied_count: usize,
    output: *mut ThoulessComplex64,
    output_capacity: usize,
) -> ThoulessStatus {
    boundary(|| {
        let hamiltonian = unsafe { read_complex_matrix(hamiltonian, "Hamiltonian")? };
        let derivatives = unsafe { read_complex_tensor3(derivatives, "Hamiltonian derivatives")? };
        let occupied =
            unsafe { borrowed_slice(occupied_states, occupied_count, "occupied states")? };
        let tensor = quantum_geometric_tensor_from_hamiltonian_derivatives(
            &hamiltonian,
            &derivatives,
            occupied,
        )
        .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        let required = tensor
            .direction_count()
            .checked_mul(tensor.direction_count())
            .and_then(|value| value.checked_mul(tensor.occupied_count()))
            .and_then(|value| value.checked_mul(tensor.occupied_count()))
            .ok_or_else(|| {
                AbiError::new(
                    ThoulessStatus::ResourceExhausted,
                    "quantum-geometric tensor size overflow",
                )
            })?;
        if output_capacity < required {
            return Err(AbiError::new(
                ThoulessStatus::BufferTooSmall,
                format!("tensor output has {output_capacity} elements; required {required}"),
            ));
        }
        let output = unsafe { borrowed_mut_slice(output, output_capacity, "tensor output")? };
        let mut index = 0;
        for first in 0..tensor.direction_count() {
            for second in 0..tensor.direction_count() {
                for value in tensor
                    .component(first, second)
                    .expect("validated component")
                    .as_slice()
                {
                    output[index] = (*value).into();
                    index += 1;
                }
            }
        }
        Ok(())
    })
}

/// Compute the real-space local Chern marker of a finite Hamiltonian.
#[no_mangle]
pub unsafe extern "C" fn thouless_local_chern_marker(
    hamiltonian: ThoulessC64MatrixView,
    positions: ThoulessF64MatrixView,
    occupied_states: *const usize,
    occupied_count: usize,
    cell_area: f64,
    output: *mut f64,
    output_capacity: usize,
) -> ThoulessStatus {
    boundary(|| {
        let hamiltonian = unsafe { read_complex_matrix(hamiltonian, "Hamiltonian")? };
        let positions = unsafe { read_real_matrix(positions, "positions")? };
        if positions.columns() != 2 {
            return Err(AbiError::shape("positions must have two columns"));
        }
        let positions = real_rows(&positions)
            .into_iter()
            .map(|position| [position[0], position[1]])
            .collect::<Vec<_>>();
        let occupied =
            unsafe { borrowed_slice(occupied_states, occupied_count, "occupied states")? };
        let marker =
            local_chern_marker_from_hamiltonian(&hamiltonian, &positions, occupied, cell_area)
                .map_err(|error| AbiError::invalid(error.to_string()))?;
        write_f64(&marker, output, output_capacity, "local Chern marker")
    })
}

/// Project sampled state frames onto localized trial orbitals.
#[no_mangle]
pub unsafe extern "C" fn thouless_wannier_project_trials(
    frames: ThoulessC64Tensor3View,
    trials: ThoulessC64MatrixView,
    singular_tolerance: f64,
    output: ThoulessC64Tensor3Mut,
) -> ThoulessStatus {
    boundary(|| {
        let frames = unsafe { read_complex_tensor3(frames, "sampled frames")? };
        let trials = unsafe { read_complex_matrix(trials, "trial orbitals")? };
        let projected = project_trials(&frames, &trials, singular_tolerance)
            .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        unsafe { write_complex_tensor3(&projected, output, "projected frames") }
    })
}

/// Compute gauge-invariant occupation-weighted intrinsic curvature of a model.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_intrinsic_curvature(
    model: *const ThoulessModel,
    momentum: *const f64,
    momentum_length: usize,
    chemical_potential: f64,
    temperature: f64,
    cartesian: bool,
    degeneracy_tolerance: f64,
    output: ThoulessF64MatrixMut,
) -> ThoulessStatus {
    boundary(|| {
        let model = unsafe { model_ref(model)? };
        let momentum = unsafe { borrowed_slice(momentum, momentum_length, "momentum")? };
        let fermi = FermiDistribution::new(chemical_potential, temperature)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        let coordinates = if cartesian {
            MomentumCoordinates::Cartesian
        } else {
            MomentumCoordinates::Reduced
        };
        let curvature = intrinsic_berry_curvature_from_model(
            &model.inner,
            momentum,
            fermi,
            coordinates,
            degeneracy_tolerance,
        )
        .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        unsafe { write_real_matrix(&curvature, output, "intrinsic curvature") }
    })
}

/// Project a diagonal observable into a state frame.
#[no_mangle]
pub unsafe extern "C" fn thouless_project_diagonal_observable(
    states: ThoulessC64MatrixView,
    diagonal: *const f64,
    diagonal_length: usize,
    output: ThoulessC64MatrixMut,
) -> ThoulessStatus {
    boundary(|| {
        let states = unsafe { read_complex_matrix(states, "states")? };
        let diagonal = unsafe { borrowed_slice(diagonal, diagonal_length, "diagonal")? };
        let projected = project_diagonal_observable(&states, diagonal)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe { write_complex_matrix(&projected, output, "projected observable") }
    })
}

fn read_leads(
    leads: *const ThoulessLeadView,
    lead_count: usize,
) -> Result<Vec<LeadContact>, AbiError> {
    let leads = unsafe { borrowed_slice(leads, lead_count, "leads")? };
    leads
        .iter()
        .map(|lead| {
            LeadContact::new(
                unsafe { read_complex_matrix(lead.cell_hamiltonian, "lead cell")? },
                unsafe { read_complex_matrix(lead.inter_cell_hopping, "lead hopping")? },
                unsafe { read_complex_matrix(lead.coupling, "lead coupling")? },
            )
            .map_err(|error| AbiError::invalid(error.to_string()))
        })
        .collect()
}

/// Solve a general multi-terminal coherent scattering problem.
#[no_mangle]
pub unsafe extern "C" fn thouless_open_system_transmissions(
    device_hamiltonian: ThoulessC64MatrixView,
    leads: *const ThoulessLeadView,
    lead_count: usize,
    energy: f64,
    broadening: f64,
    output: ThoulessF64MatrixMut,
) -> ThoulessStatus {
    boundary(|| {
        let device = unsafe { read_complex_matrix(device_hamiltonian, "device Hamiltonian")? };
        let leads = read_leads(leads, lead_count)?;
        let options = SurfaceGreenOptions {
            broadening,
            ..SurfaceGreenOptions::default()
        };
        let solution = solve_open_system(&device, &leads, energy, options)
            .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        let transmissions = solution
            .transmission_matrix()
            .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        let matrix = RealMatrix::new(
            lead_count,
            lead_count,
            transmissions.into_iter().flatten().collect(),
        )
        .map_err(|error| AbiError::shape(error.to_string()))?;
        unsafe { write_real_matrix(&matrix, output, "transmission matrix") }
    })
}

fn computed_modes(
    cell_hamiltonian: ThoulessC64MatrixView,
    inter_cell_hopping: ThoulessC64MatrixView,
) -> Result<thouless::lead_modes::PropagatingLeadModes, AbiError> {
    propagating_modes(
        &unsafe { read_complex_matrix(cell_hamiltonian, "lead cell Hamiltonian")? },
        &unsafe { read_complex_matrix(inter_cell_hopping, "lead hopping")? },
    )
    .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))
}

/// Query propagating mode and incoming mode counts.
#[no_mangle]
pub unsafe extern "C" fn thouless_lead_mode_count(
    cell_hamiltonian: ThoulessC64MatrixView,
    inter_cell_hopping: ThoulessC64MatrixView,
    mode_count: *mut usize,
    incoming_count: *mut usize,
) -> ThoulessStatus {
    boundary(|| {
        if mode_count.is_null() || incoming_count.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "lead mode count output is null",
            ));
        }
        let modes = computed_modes(cell_hamiltonian, inter_cell_hopping)?;
        unsafe {
            *mode_count = modes.velocities().len();
            *incoming_count = modes.incoming_count();
        }
        Ok(())
    })
}

/// Return propagating wave functions, velocities, and momenta.
#[no_mangle]
pub unsafe extern "C" fn thouless_lead_modes(
    cell_hamiltonian: ThoulessC64MatrixView,
    inter_cell_hopping: ThoulessC64MatrixView,
    wave_functions: ThoulessC64MatrixMut,
    velocities: *mut f64,
    velocity_capacity: usize,
    momenta: *mut f64,
    momentum_capacity: usize,
) -> ThoulessStatus {
    boundary(|| {
        let modes = computed_modes(cell_hamiltonian, inter_cell_hopping)?;
        unsafe {
            write_complex_matrix(
                modes.wave_functions(),
                wave_functions,
                "lead wave functions",
            )?
        };
        write_f64(
            modes.velocities(),
            velocities,
            velocity_capacity,
            "lead velocities",
        )?;
        write_f64(modes.momenta(), momenta, momentum_capacity, "lead momenta")
    })
}

/// Compute and causally regularize a periodic lead self-energy.
#[no_mangle]
pub unsafe extern "C" fn thouless_lead_self_energy(
    cell_hamiltonian: ThoulessC64MatrixView,
    inter_cell_hopping: ThoulessC64MatrixView,
    energy: f64,
    broadening: f64,
    maximum_rank: usize,
    use_maximum_rank: bool,
    output: ThoulessC64MatrixMut,
) -> ThoulessStatus {
    boundary(|| {
        let cell = unsafe { read_complex_matrix(cell_hamiltonian, "lead cell Hamiltonian")? };
        let hopping = unsafe { read_complex_matrix(inter_cell_hopping, "lead hopping")? };
        let options = SurfaceGreenOptions {
            broadening,
            ..SurfaceGreenOptions::default()
        };
        let self_energy = retarded_lead_self_energy(&cell, &hopping, energy, options)
            .and_then(|value| {
                regularize_retarded_self_energy(&value, use_maximum_rank.then_some(maximum_rank))
            })
            .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        unsafe { write_complex_matrix(&self_energy, output, "lead self-energy") }
    })
}

/// Validate one Hamiltonian against a chiral symmetry.
#[no_mangle]
pub unsafe extern "C" fn thouless_validate_chiral_symmetry(
    matrix: ThoulessC64MatrixView,
    chiral: ThoulessC64MatrixView,
    violation_count: *mut usize,
) -> ThoulessStatus {
    boundary(|| {
        if violation_count.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "violation count is null",
            ));
        }
        let symmetry = DiscreteSymmetry::new(
            None,
            None,
            None,
            Some(unsafe { read_complex_matrix(chiral, "chiral symmetry")? }),
        )
        .map_err(|error| AbiError::invalid(error.to_string()))?;
        let violations = symmetry
            .validate(&unsafe { read_complex_matrix(matrix, "Hamiltonian")? })
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe { *violation_count = violations.len() };
        Ok(())
    })
}

/// Construct a particle-hole-symmetric basis and its stable ordering.
#[no_mangle]
pub unsafe extern "C" fn thouless_particle_hole_basis(
    wave_functions: ThoulessC64MatrixView,
    particle_hole: ThoulessC64MatrixView,
    output: ThoulessC64MatrixMut,
    ordering: *mut usize,
    ordering_capacity: usize,
) -> ThoulessStatus {
    boundary(|| {
        let result = particle_hole_symmetric_basis(
            &unsafe { read_complex_matrix(wave_functions, "wave functions")? },
            &unsafe { read_complex_matrix(particle_hole, "particle-hole symmetry")? },
        )
        .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        unsafe { write_complex_matrix(result.wave_functions(), output, "particle-hole basis")? };
        write_usize(
            result.ordering(),
            ordering,
            ordering_capacity,
            "particle-hole ordering",
        )
    })
}

/// Project caller-supplied components into an Altland-Zirnbauer Gaussian ensemble.
#[no_mangle]
pub unsafe extern "C" fn thouless_random_gaussian_matrix(
    dimension: usize,
    symmetry: u32,
    variance: f64,
    real_components: *const f64,
    real_count: usize,
    imaginary_components: *const f64,
    imaginary_count: usize,
    output: ThoulessC64MatrixMut,
) -> ThoulessStatus {
    boundary(|| {
        let real = unsafe { borrowed_slice(real_components, real_count, "real components")? };
        let imaginary = unsafe {
            borrowed_slice(
                imaginary_components,
                imaginary_count,
                "imaginary components",
            )?
        };
        let matrix = gaussian_from_components(
            dimension,
            symmetry_class(symmetry)?,
            variance,
            real,
            imaginary,
        )
        .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe { write_complex_matrix(&matrix, output, "random matrix") }
    })
}

/// Return one deterministic uniform random-access variate.
#[no_mangle]
pub unsafe extern "C" fn thouless_digest_uniform(
    input: *const u8,
    input_length: usize,
    salt: *const u8,
    salt_length: usize,
    output: *mut f64,
) -> ThoulessStatus {
    boundary(|| {
        if output.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "uniform variate output is null",
            ));
        }
        let input = unsafe { borrowed_slice(input, input_length, "digest input")? };
        let salt = unsafe { borrowed_slice(salt, salt_length, "digest salt")? };
        unsafe { *output = uniform(input, salt) };
        Ok(())
    })
}

/// Return one deterministic Gaussian random-access variate.
#[no_mangle]
pub unsafe extern "C" fn thouless_digest_gaussian(
    input: *const u8,
    input_length: usize,
    salt: *const u8,
    salt_length: usize,
    output: *mut f64,
) -> ThoulessStatus {
    boundary(|| {
        if output.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "Gaussian variate output is null",
            ));
        }
        let input = unsafe { borrowed_slice(input, input_length, "digest input")? };
        let salt = unsafe { borrowed_slice(salt, salt_length, "digest salt")? };
        unsafe { *output = digest_gaussian(input, salt) };
        Ok(())
    })
}

/// LLL-reduce a full-rank basis and return its integer transformation.
#[no_mangle]
pub unsafe extern "C" fn thouless_lll_reduce(
    basis: ThoulessF64MatrixView,
    reduction_parameter: f64,
    reduced_basis: ThoulessF64MatrixMut,
    transformation: *mut i64,
    transformation_capacity: usize,
) -> ThoulessStatus {
    boundary(|| {
        let basis = unsafe { read_real_matrix(basis, "lattice basis")? };
        let reduced = lll_reduce(&real_rows(&basis), reduction_parameter)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        let reduced_matrix = RealMatrix::new(
            reduced.vectors().len(),
            reduced.vectors().first().map_or(0, Vec::len),
            reduced.vectors().iter().flatten().copied().collect(),
        )
        .map_err(|error| AbiError::shape(error.to_string()))?;
        unsafe { write_real_matrix(&reduced_matrix, reduced_basis, "reduced basis")? };
        let flat = reduced
            .transformation()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        if transformation_capacity < flat.len() {
            return Err(AbiError::new(
                ThoulessStatus::BufferTooSmall,
                format!(
                    "transformation output has {transformation_capacity} elements; required {}",
                    flat.len()
                ),
            ));
        }
        let output = unsafe {
            borrowed_mut_slice(
                transformation,
                transformation_capacity,
                "LLL transformation",
            )?
        };
        output[..flat.len()].copy_from_slice(&flat);
        Ok(())
    })
}

/// Compress a directed graph into canonical outgoing CSR adjacency.
#[no_mangle]
pub unsafe extern "C" fn thouless_graph_compress(
    node_count: usize,
    tails: *const i64,
    heads: *const i64,
    edge_count: usize,
    row_offsets: *mut usize,
    row_offset_capacity: usize,
    neighbors: *mut i64,
    neighbor_capacity: usize,
) -> ThoulessStatus {
    boundary(|| {
        let tails = unsafe { borrowed_slice(tails, edge_count, "edge tails")? };
        let heads = unsafe { borrowed_slice(heads, edge_count, "edge heads")? };
        let mut builder = DirectedGraphBuilder::new();
        builder
            .set_node_count(node_count)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        builder
            .extend_edges(
                tails
                    .iter()
                    .copied()
                    .zip(heads.iter().copied())
                    .map(|(tail, head)| DirectedEdge::new(tail, head)),
            )
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        let graph = builder
            .compress(CompressionOptions::default())
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        let mut offsets = Vec::with_capacity(node_count + 1);
        let mut flat_neighbors = Vec::with_capacity(graph.outgoing_edge_count());
        offsets.push(0);
        for node in 0..node_count {
            flat_neighbors.extend_from_slice(
                graph
                    .outgoing_neighbors(node as i64)
                    .map_err(|error| AbiError::invalid(error.to_string()))?,
            );
            offsets.push(flat_neighbors.len());
        }
        write_usize(
            &offsets,
            row_offsets,
            row_offset_capacity,
            "graph row offsets",
        )?;
        if neighbor_capacity < flat_neighbors.len() {
            return Err(AbiError::new(
                ThoulessStatus::BufferTooSmall,
                format!(
                    "graph neighbor output has {neighbor_capacity} elements; required {}",
                    flat_neighbors.len()
                ),
            ));
        }
        let output =
            unsafe { borrowed_mut_slice(neighbors, neighbor_capacity, "graph neighbors")? };
        output[..flat_neighbors.len()].copy_from_slice(&flat_neighbors);
        Ok(())
    })
}

/// Compute a complex Schur decomposition.
#[no_mangle]
pub unsafe extern "C" fn thouless_dense_schur(
    matrix: ThoulessC64MatrixView,
    form: ThoulessC64MatrixMut,
    vectors: ThoulessC64MatrixMut,
    eigenvalues: *mut ThoulessComplex64,
    eigenvalue_capacity: usize,
) -> ThoulessStatus {
    boundary(|| {
        let decomposition = schur(&unsafe { read_complex_matrix(matrix, "Schur matrix")? })
            .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        unsafe {
            write_complex_matrix(decomposition.form(), form, "Schur form")?;
            write_complex_matrix(decomposition.vectors(), vectors, "Schur vectors")?;
        }
        if eigenvalue_capacity < decomposition.eigenvalues().len() {
            return Err(AbiError::new(
                ThoulessStatus::BufferTooSmall,
                "Schur eigenvalue output is too small",
            ));
        }
        let output =
            unsafe { borrowed_mut_slice(eigenvalues, eigenvalue_capacity, "Schur eigenvalues")? };
        for (destination, value) in output.iter_mut().zip(decomposition.eigenvalues()) {
            *destination = (*value).into();
        }
        Ok(())
    })
}

/// Factor and solve a canonical complex CSR matrix.
#[no_mangle]
pub unsafe extern "C" fn thouless_sparse_solve(
    rows: usize,
    columns: usize,
    row_offsets: *const usize,
    row_offset_count: usize,
    column_indices: *const usize,
    values: *const ThoulessComplex64,
    nonzero_count: usize,
    right_hand_side: ThoulessC64MatrixView,
    output: ThoulessC64MatrixMut,
) -> ThoulessStatus {
    boundary(|| {
        let row_offsets =
            unsafe { borrowed_slice(row_offsets, row_offset_count, "CSR row offsets")? };
        let column_indices =
            unsafe { borrowed_slice(column_indices, nonzero_count, "CSR column indices")? };
        let values = unsafe { borrowed_slice(values, nonzero_count, "CSR values")? };
        let matrix = CsrMatrix::new(
            rows,
            columns,
            row_offsets.to_vec(),
            column_indices.to_vec(),
            values.iter().copied().map(Complex64::from).collect(),
        )
        .map_err(|error| AbiError::shape(error.to_string()))?;
        let factorization = SparseLuFactorization::factor(&matrix)
            .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        let solution = factorization
            .solve(&unsafe { read_complex_matrix(right_hand_side, "right-hand side")? })
            .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        unsafe { write_complex_matrix(&solution, output, "sparse solution") }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ThoulessC64MatrixMut, ThoulessC64MatrixView};

    fn view(values: &[ThoulessComplex64], rows: usize, columns: usize) -> ThoulessC64MatrixView {
        ThoulessC64MatrixView {
            data: values.as_ptr(),
            rows,
            columns,
            row_stride: columns,
            column_stride: 1,
        }
    }

    fn output(
        values: &mut [ThoulessComplex64],
        rows: usize,
        columns: usize,
    ) -> ThoulessC64MatrixMut {
        ThoulessC64MatrixMut {
            data: values.as_mut_ptr(),
            rows,
            columns,
            row_stride: columns,
            column_stride: 1,
        }
    }

    #[test]
    fn dense_and_sparse_abi_paths_share_scientific_results() {
        // SAFETY: all ABI views and pointers refer to live exact-size arrays.
        unsafe {
            let matrix = [
                ThoulessComplex64 { re: 2.0, im: 0.0 },
                ThoulessComplex64 { re: 0.0, im: 1.0 },
                ThoulessComplex64 { re: 0.0, im: -1.0 },
                ThoulessComplex64 { re: 3.0, im: 0.0 },
            ];
            let rhs = [
                ThoulessComplex64 { re: 1.0, im: 0.0 },
                ThoulessComplex64 { re: 2.0, im: 0.0 },
            ];
            let mut solution = [ThoulessComplex64::default(); 2];
            assert_eq!(
                thouless_sparse_solve(
                    2,
                    2,
                    [0usize, 2, 4].as_ptr(),
                    3,
                    [0usize, 1, 0, 1].as_ptr(),
                    matrix.as_ptr(),
                    4,
                    view(&rhs, 2, 1),
                    output(&mut solution, 2, 1),
                ),
                ThoulessStatus::Success
            );
            let first = Complex64::from(solution[0]);
            let second = Complex64::from(solution[1]);
            assert!(
                (Complex64::new(2.0, 0.0) * first + Complex64::i() * second
                    - Complex64::new(1.0, 0.0))
                .norm()
                    <= 1.0e-10
            );
        }
    }

    #[test]
    fn digest_and_bloch_phase_are_deterministic() {
        // SAFETY: all ABI pointers refer to live exact-size arrays or scalars.
        unsafe {
            let mut value = 0.0;
            assert_eq!(
                thouless_digest_uniform(b"input".as_ptr(), 5, b"salt".as_ptr(), 4, &mut value),
                ThoulessStatus::Success
            );
            assert_eq!(value, uniform(b"input", b"salt"));
            let mut phase = ThoulessComplex64::default();
            assert_eq!(
                thouless_bloch_phase([1i64].as_ptr(), 1, [0.25].as_ptr(), 1, &mut phase),
                ThoulessStatus::Success
            );
            assert!((phase.re - 0.25_f64.cos()).abs() <= 1.0e-12);
            assert!((phase.im - 0.25_f64.sin()).abs() <= 1.0e-12);
        }
    }
}
