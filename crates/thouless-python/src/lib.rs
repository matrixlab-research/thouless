use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use thouless::differentiation::{finite_difference_uniform, DifferenceScheme};
use thouless::geometry::ReciprocalPath;
use thouless::lattice_reduction::{
    closest_lattice_vectors, gram_schmidt, gram_schmidt_coefficient, is_c_reduced, lll_reduce,
    voronoi_neighbors,
};
use thouless::model::{ModelBuilder, OrbitalId, TightBindingModel};
use thouless::observables::{pauli_coefficients, project_diagonal_observable};
use thouless::random_matrix::{circular_from_components, gaussian_from_components, SymmetryClass};
use thouless::spectrum::hermitian_eigensystem;
use thouless::symmetry::DiscreteSymmetry as NativeDiscreteSymmetry;
use thouless::topology::{
    chern_numbers_on_uniform_grid, connection_from_link, parallel_transport_link, plaquette_flux,
    second_chern_from_hamiltonian_derivatives, wilson_line_phase, wilson_loop_eigenphases,
};
use thouless::transform::{change_nonperiodic_vector, make_supercell, remove_orbitals};
use thouless::transport::{solve_open_system, LeadContact, SurfaceGreenOptions};
use thouless::{Complex64, ComplexMatrix};

type HoppingInput = (usize, usize, Vec<i32>, Vec<Vec<Complex64>>);
type LeadInput = (
    Vec<Vec<Complex64>>,
    Vec<Vec<Complex64>>,
    Vec<Vec<Complex64>>,
);
type OpenSystemOutput = (
    Vec<Vec<Complex64>>,
    Vec<Vec<Vec<Complex64>>>,
    Vec<Vec<Vec<Complex64>>>,
    Vec<Vec<f64>>,
);
type ModelOutput = (
    Vec<Vec<f64>>,
    Vec<usize>,
    Vec<Vec<f64>>,
    Vec<usize>,
    Vec<Vec<Vec<Complex64>>>,
    Vec<HoppingInput>,
);
type ReciprocalPathOutput = (Vec<Vec<f64>>, Vec<f64>, Vec<f64>);
type SupercellOutput = (ModelOutput, Vec<Vec<i32>>);
type MatrixRows = Vec<Vec<Complex64>>;
type LatticeReductionOutput = (Vec<Vec<f64>>, Vec<Vec<i64>>);
type DiscreteSymmetryOutput = (
    Option<Vec<MatrixRows>>,
    Option<MatrixRows>,
    Option<MatrixRows>,
    Option<MatrixRows>,
);

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn symmetry_class(name: &str) -> PyResult<SymmetryClass> {
    match name {
        "A" => Ok(SymmetryClass::A),
        "AI" => Ok(SymmetryClass::Ai),
        "AII" => Ok(SymmetryClass::Aii),
        "AIII" => Ok(SymmetryClass::Aiii),
        "BDI" => Ok(SymmetryClass::Bdi),
        "CII" => Ok(SymmetryClass::Cii),
        "D" => Ok(SymmetryClass::D),
        "DIII" => Ok(SymmetryClass::Diii),
        "C" => Ok(SymmetryClass::C),
        "CI" => Ok(SymmetryClass::Ci),
        _ => Err(PyValueError::new_err("unknown Altland-Zirnbauer class")),
    }
}

fn matrix_from_rows(rows: Vec<Vec<Complex64>>) -> PyResult<ComplexMatrix> {
    let row_count = rows.len();
    let column_count = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != column_count) {
        return Err(PyValueError::new_err(
            "matrix rows must all have the same length",
        ));
    }
    let data = rows.into_iter().flatten().collect();
    ComplexMatrix::new(row_count, column_count, data).map_err(value_error)
}

fn matrix_to_rows(matrix: &ComplexMatrix) -> Vec<Vec<Complex64>> {
    matrix
        .as_slice()
        .chunks(matrix.columns())
        .map(<[Complex64]>::to_vec)
        .collect()
}

fn optional_matrix(rows: Option<MatrixRows>) -> PyResult<Option<ComplexMatrix>> {
    rows.map(matrix_from_rows).transpose()
}

fn build_model(
    primitive_vectors: Vec<Vec<f64>>,
    periodic_axes: Vec<usize>,
    orbital_positions: Vec<Vec<f64>>,
    degrees_of_freedom: Vec<usize>,
    onsite_blocks: Vec<Vec<Vec<Complex64>>>,
    hoppings: Vec<HoppingInput>,
) -> PyResult<TightBindingModel> {
    let orbital_count = orbital_positions.len();
    if degrees_of_freedom.len() != orbital_count || onsite_blocks.len() != orbital_count {
        return Err(PyValueError::new_err(
            "orbital positions, degrees of freedom, and onsite blocks must have equal lengths",
        ));
    }

    let lattice =
        thouless::model::Lattice::new(primitive_vectors, periodic_axes).map_err(value_error)?;
    let mut builder = ModelBuilder::new(lattice);
    let mut orbital_ids = Vec::with_capacity(orbital_count);
    for (index, (position, degrees)) in orbital_positions
        .into_iter()
        .zip(degrees_of_freedom)
        .enumerate()
    {
        orbital_ids.push(
            builder
                .add_orbital_with_dof(format!("orbital-{index}"), position, degrees)
                .map_err(value_error)?,
        );
    }

    for (orbital, rows) in orbital_ids.iter().copied().zip(onsite_blocks) {
        builder
            .set_onsite_block(orbital, matrix_from_rows(rows)?)
            .map_err(value_error)?;
    }
    for (target, source, offset, rows) in hoppings {
        let target = orbital_id(&orbital_ids, target)?;
        let source = orbital_id(&orbital_ids, source)?;
        builder
            .add_hopping_block(target, source, offset, matrix_from_rows(rows)?)
            .map_err(value_error)?;
    }
    builder.build().map_err(value_error)
}

fn orbital_id(orbitals: &[OrbitalId], index: usize) -> PyResult<OrbitalId> {
    orbitals
        .get(index)
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("unknown orbital index {index}")))
}

fn model_to_output(model: &TightBindingModel) -> ModelOutput {
    (
        model.lattice().primitive_vectors().to_vec(),
        model.lattice().periodic_axes().to_vec(),
        model
            .orbitals()
            .iter()
            .map(|orbital| orbital.reduced_position().to_vec())
            .collect(),
        model
            .orbitals()
            .iter()
            .map(thouless::model::Orbital::degrees_of_freedom)
            .collect(),
        model.onsite_blocks().iter().map(matrix_to_rows).collect(),
        model
            .hoppings()
            .iter()
            .map(|hopping| {
                (
                    hopping.target().index(),
                    hopping.source().index(),
                    hopping.cell_offset().to_vec(),
                    matrix_to_rows(hopping.amplitude()),
                )
            })
            .collect(),
    )
}

#[pyfunction]
fn hamiltonian(
    primitive_vectors: Vec<Vec<f64>>,
    periodic_axes: Vec<usize>,
    orbital_positions: Vec<Vec<f64>>,
    degrees_of_freedom: Vec<usize>,
    onsite_blocks: Vec<Vec<Vec<Complex64>>>,
    hoppings: Vec<HoppingInput>,
    momentum: Vec<f64>,
) -> PyResult<Vec<Vec<Complex64>>> {
    let model = build_model(
        primitive_vectors,
        periodic_axes,
        orbital_positions,
        degrees_of_freedom,
        onsite_blocks,
        hoppings,
    )?;
    model
        .hamiltonian(&momentum)
        .map(|matrix| matrix_to_rows(&matrix))
        .map_err(value_error)
}

#[pyfunction]
fn eigensystem(
    primitive_vectors: Vec<Vec<f64>>,
    periodic_axes: Vec<usize>,
    orbital_positions: Vec<Vec<f64>>,
    degrees_of_freedom: Vec<usize>,
    onsite_blocks: Vec<Vec<Vec<Complex64>>>,
    hoppings: Vec<HoppingInput>,
    momentum: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<Vec<Complex64>>)> {
    let model = build_model(
        primitive_vectors,
        periodic_axes,
        orbital_positions,
        degrees_of_freedom,
        onsite_blocks,
        hoppings,
    )?;
    let solution = model.eigensystem(&momentum).map_err(value_error)?;
    Ok((
        solution.eigenvalues().to_vec(),
        matrix_to_rows(solution.eigenvectors()),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn remove_model_orbitals(
    primitive_vectors: Vec<Vec<f64>>,
    periodic_axes: Vec<usize>,
    orbital_positions: Vec<Vec<f64>>,
    degrees_of_freedom: Vec<usize>,
    onsite_blocks: Vec<Vec<Vec<Complex64>>>,
    hoppings: Vec<HoppingInput>,
    removed: Vec<usize>,
) -> PyResult<ModelOutput> {
    let model = build_model(
        primitive_vectors,
        periodic_axes,
        orbital_positions,
        degrees_of_freedom,
        onsite_blocks,
        hoppings,
    )?;
    remove_orbitals(&model, &removed)
        .map(|transformed| model_to_output(&transformed))
        .map_err(value_error)
}

#[pyfunction]
#[pyo3(signature = (
    primitive_vectors,
    periodic_axes,
    orbital_positions,
    degrees_of_freedom,
    onsite_blocks,
    hoppings,
    direction,
    move_periodic_to_home,
    replacement=None
))]
#[allow(clippy::too_many_arguments)]
fn change_model_nonperiodic_vector(
    primitive_vectors: Vec<Vec<f64>>,
    periodic_axes: Vec<usize>,
    orbital_positions: Vec<Vec<f64>>,
    degrees_of_freedom: Vec<usize>,
    onsite_blocks: Vec<Vec<Vec<Complex64>>>,
    hoppings: Vec<HoppingInput>,
    direction: usize,
    move_periodic_to_home: bool,
    replacement: Option<Vec<f64>>,
) -> PyResult<ModelOutput> {
    let model = build_model(
        primitive_vectors,
        periodic_axes,
        orbital_positions,
        degrees_of_freedom,
        onsite_blocks,
        hoppings,
    )?;
    change_nonperiodic_vector(
        &model,
        direction,
        replacement.as_deref(),
        move_periodic_to_home,
    )
    .map(|transformed| model_to_output(&transformed))
    .map_err(value_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn make_model_supercell(
    primitive_vectors: Vec<Vec<f64>>,
    periodic_axes: Vec<usize>,
    orbital_positions: Vec<Vec<f64>>,
    degrees_of_freedom: Vec<usize>,
    onsite_blocks: Vec<Vec<Vec<Complex64>>>,
    hoppings: Vec<HoppingInput>,
    integer_basis: Vec<Vec<i32>>,
    move_periodic_to_home: bool,
) -> PyResult<SupercellOutput> {
    let model = build_model(
        primitive_vectors,
        periodic_axes,
        orbital_positions,
        degrees_of_freedom,
        onsite_blocks,
        hoppings,
    )?;
    make_supercell(&model, &integer_basis, move_periodic_to_home)
        .map(|result| {
            (
                model_to_output(result.model()),
                result.translations().to_vec(),
            )
        })
        .map_err(value_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uniform_grid_chern(
    primitive_vectors: Vec<Vec<f64>>,
    periodic_axes: Vec<usize>,
    orbital_positions: Vec<Vec<f64>>,
    degrees_of_freedom: Vec<usize>,
    onsite_blocks: Vec<Vec<Vec<Complex64>>>,
    hoppings: Vec<HoppingInput>,
    samples: Vec<usize>,
    plane: Vec<usize>,
    occupied_states: Vec<usize>,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    if plane.len() != 2 {
        return Err(PyValueError::new_err(
            "Chern plane must contain exactly two directions",
        ));
    }
    let model = build_model(
        primitive_vectors,
        periodic_axes,
        orbital_positions,
        degrees_of_freedom,
        onsite_blocks,
        hoppings,
    )?;
    let result =
        chern_numbers_on_uniform_grid(&model, &samples, [plane[0], plane[1]], &occupied_states)
            .map_err(value_error)?;
    Ok((result.values().to_vec(), result.spectator_shape().to_vec()))
}

#[pyfunction]
fn second_chern_kubo(
    hamiltonians: Vec<Vec<Vec<Complex64>>>,
    derivatives: Vec<Vec<Vec<Vec<Complex64>>>>,
    grid_shape: Vec<usize>,
    coordinate_steps: Vec<f64>,
    fourth_axis_periodic: bool,
    occupied_states: Vec<usize>,
) -> PyResult<(Vec<f64>, f64)> {
    let hamiltonians = hamiltonians
        .into_iter()
        .map(matrix_from_rows)
        .collect::<PyResult<Vec<_>>>()?;
    let derivatives = derivatives
        .into_iter()
        .map(|group| {
            let matrices = group
                .into_iter()
                .map(matrix_from_rows)
                .collect::<PyResult<Vec<_>>>()?;
            matrices.try_into().map_err(|matrices: Vec<_>| {
                PyValueError::new_err(format!(
                    "each grid point requires four Hamiltonian derivatives; received {}",
                    matrices.len()
                ))
            })
        })
        .collect::<PyResult<Vec<[ComplexMatrix; 4]>>>()?;
    let result = second_chern_from_hamiltonian_derivatives(
        &hamiltonians,
        &derivatives,
        &grid_shape,
        &coordinate_steps,
        fourth_axis_periodic,
        &occupied_states,
    )
    .map_err(value_error)?;
    Ok((result.slice_densities().to_vec(), result.value()))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn momentum_derivatives(
    primitive_vectors: Vec<Vec<f64>>,
    periodic_axes: Vec<usize>,
    orbital_positions: Vec<Vec<f64>>,
    degrees_of_freedom: Vec<usize>,
    onsite_blocks: Vec<Vec<Vec<Complex64>>>,
    hoppings: Vec<HoppingInput>,
    momentum: Vec<f64>,
    cartesian: bool,
) -> PyResult<Vec<Vec<Vec<Complex64>>>> {
    let model = build_model(
        primitive_vectors,
        periodic_axes,
        orbital_positions,
        degrees_of_freedom,
        onsite_blocks,
        hoppings,
    )?;
    let derivatives = if cartesian {
        model.cartesian_momentum_derivatives(&momentum)
    } else {
        model.reduced_momentum_derivatives(&momentum)
    }
    .map_err(value_error)?;
    Ok(derivatives.iter().map(matrix_to_rows).collect::<Vec<_>>())
}

#[pyfunction]
fn finite_difference(
    samples: Vec<Vec<Vec<Complex64>>>,
    step: f64,
    periodic: bool,
    scheme: &str,
) -> PyResult<Vec<Vec<Vec<Complex64>>>> {
    let scheme = match scheme {
        "central" => DifferenceScheme::Central,
        "forward" => DifferenceScheme::Forward,
        _ => {
            return Err(PyValueError::new_err(
                "finite-difference scheme must be 'central' or 'forward'",
            ))
        }
    };
    let matrices = samples
        .into_iter()
        .map(matrix_from_rows)
        .collect::<PyResult<Vec<_>>>()?;
    finite_difference_uniform(&matrices, step, periodic, scheme)
        .map(|result| result.iter().map(matrix_to_rows).collect())
        .map_err(value_error)
}

#[pyfunction(signature = (device_hamiltonian, leads, energy, broadening=None))]
fn open_system_transmissions(
    device_hamiltonian: Vec<Vec<Complex64>>,
    leads: Vec<LeadInput>,
    energy: f64,
    broadening: Option<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let device_hamiltonian = matrix_from_rows(device_hamiltonian)?;
    let leads = leads
        .into_iter()
        .map(|(cell, hopping, coupling)| {
            LeadContact::new(
                matrix_from_rows(cell)?,
                matrix_from_rows(hopping)?,
                matrix_from_rows(coupling)?,
            )
            .map_err(value_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut options = SurfaceGreenOptions::default();
    if let Some(broadening) = broadening {
        options.broadening = broadening;
    }
    let solution =
        solve_open_system(&device_hamiltonian, &leads, energy, options).map_err(value_error)?;
    (0..leads.len())
        .map(|drain| {
            (0..leads.len())
                .map(|source| solution.transmission(drain, source).map_err(value_error))
                .collect()
        })
        .collect()
}

#[pyfunction(signature = (device_hamiltonian, leads, energy, broadening=None))]
fn open_system_solution(
    device_hamiltonian: Vec<Vec<Complex64>>,
    leads: Vec<LeadInput>,
    energy: f64,
    broadening: Option<f64>,
) -> PyResult<OpenSystemOutput> {
    let device_hamiltonian = matrix_from_rows(device_hamiltonian)?;
    let leads = leads
        .into_iter()
        .map(|(cell, hopping, coupling)| {
            LeadContact::new(
                matrix_from_rows(cell)?,
                matrix_from_rows(hopping)?,
                matrix_from_rows(coupling)?,
            )
            .map_err(value_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut options = SurfaceGreenOptions::default();
    if let Some(broadening) = broadening {
        options.broadening = broadening;
    }
    let solution =
        solve_open_system(&device_hamiltonian, &leads, energy, options).map_err(value_error)?;
    let transmissions = (0..leads.len())
        .map(|drain| {
            (0..leads.len())
                .map(|source| solution.transmission(drain, source).map_err(value_error))
                .collect()
        })
        .collect::<PyResult<Vec<Vec<_>>>>()?;
    Ok((
        matrix_to_rows(solution.retarded_green()),
        solution
            .self_energies()
            .iter()
            .map(matrix_to_rows)
            .collect(),
        solution.broadenings().iter().map(matrix_to_rows).collect(),
        transmissions,
    ))
}

#[pyfunction]
fn wilson_phase(frames: Vec<Vec<Vec<Complex64>>>) -> PyResult<f64> {
    let frames = frames
        .into_iter()
        .map(matrix_from_rows)
        .collect::<PyResult<Vec<_>>>()?;
    wilson_line_phase(&frames).map_err(value_error)
}

#[pyfunction]
fn berry_flux(corners: Vec<Vec<Vec<Complex64>>>) -> PyResult<f64> {
    if corners.len() != 4 {
        return Err(PyValueError::new_err(
            "a plaquette requires exactly four corners",
        ));
    }
    let corners = corners
        .into_iter()
        .map(matrix_from_rows)
        .collect::<PyResult<Vec<_>>>()?;
    let corners: [ComplexMatrix; 4] = corners
        .try_into()
        .map_err(|_| PyValueError::new_err("a plaquette requires four corners"))?;
    plaquette_flux(&corners).map_err(value_error)
}

#[pyfunction]
fn transport_link(
    left: Vec<Vec<Complex64>>,
    right: Vec<Vec<Complex64>>,
) -> PyResult<Vec<Vec<Complex64>>> {
    let left = matrix_from_rows(left)?;
    let right = matrix_from_rows(right)?;
    parallel_transport_link(&left, &right)
        .map(|matrix| matrix_to_rows(&matrix))
        .map_err(value_error)
}

#[pyfunction]
fn wilson_eigenphases(frames: Vec<Vec<Vec<Complex64>>>) -> PyResult<Vec<f64>> {
    let frames = frames
        .into_iter()
        .map(matrix_from_rows)
        .collect::<PyResult<Vec<_>>>()?;
    wilson_loop_eigenphases(&frames).map_err(value_error)
}

#[pyfunction]
fn link_connection(
    link: Vec<Vec<Complex64>>,
    coordinate_step: f64,
) -> PyResult<Vec<Vec<Complex64>>> {
    let link = matrix_from_rows(link)?;
    connection_from_link(&link, coordinate_step)
        .map(|matrix| matrix_to_rows(&matrix))
        .map_err(value_error)
}

#[pyfunction]
fn diagonal_observable_matrix(
    states: Vec<Vec<Complex64>>,
    diagonal: Vec<f64>,
) -> PyResult<Vec<Vec<Complex64>>> {
    let states = matrix_from_rows(states)?;
    project_diagonal_observable(&states, &diagonal)
        .map(|matrix| matrix_to_rows(&matrix))
        .map_err(value_error)
}

#[pyfunction]
fn matrix_eigensystem(matrix: Vec<Vec<Complex64>>) -> PyResult<(Vec<f64>, Vec<Vec<Complex64>>)> {
    let matrix = matrix_from_rows(matrix)?;
    let solution = hermitian_eigensystem(&matrix, 1.0e-12).map_err(value_error)?;
    Ok((
        solution.eigenvalues().to_vec(),
        matrix_to_rows(solution.eigenvectors()),
    ))
}

#[pyfunction]
fn pauli_decompose(matrix: Vec<Vec<Complex64>>) -> PyResult<Vec<Complex64>> {
    let matrix = matrix_from_rows(matrix)?;
    pauli_coefficients(&matrix)
        .map(|values| values.to_vec())
        .map_err(value_error)
}

#[pyfunction]
fn reciprocal_path(
    primitive_vectors: Vec<Vec<f64>>,
    periodic_axes: Vec<usize>,
    nodes: Vec<Vec<f64>>,
    sample_count: usize,
) -> PyResult<ReciprocalPathOutput> {
    let lattice =
        thouless::model::Lattice::new(primitive_vectors, periodic_axes).map_err(value_error)?;
    let path = ReciprocalPath::through(&lattice, &nodes, sample_count).map_err(value_error)?;
    Ok((
        path.reduced_points().to_vec(),
        path.distances().to_vec(),
        path.node_distances().to_vec(),
    ))
}

#[pyfunction]
fn rmt_gaussian(
    dimension: usize,
    symmetry: &str,
    variance: f64,
    real: Vec<f64>,
    imaginary: Vec<f64>,
) -> PyResult<Vec<Vec<Complex64>>> {
    gaussian_from_components(
        dimension,
        symmetry_class(symmetry)?,
        variance,
        &real,
        &imaginary,
    )
    .map(|matrix| matrix_to_rows(&matrix))
    .map_err(value_error)
}

#[pyfunction(signature = (
    dimension,
    symmetry,
    real,
    imaginary,
    random_bits,
    topological_sector=None
))]
fn rmt_circular(
    dimension: usize,
    symmetry: &str,
    real: Vec<f64>,
    imaginary: Vec<f64>,
    random_bits: Vec<bool>,
    topological_sector: Option<i32>,
) -> PyResult<Vec<Vec<Complex64>>> {
    circular_from_components(
        dimension,
        symmetry_class(symmetry)?,
        topological_sector,
        &real,
        &imaginary,
        &random_bits,
    )
    .map(|matrix| matrix_to_rows(&matrix))
    .map_err(value_error)
}

#[pyfunction(signature = (
    projectors,
    time_reversal,
    particle_hole,
    chiral
))]
fn discrete_symmetry_normalize(
    projectors: Option<Vec<MatrixRows>>,
    time_reversal: Option<MatrixRows>,
    particle_hole: Option<MatrixRows>,
    chiral: Option<MatrixRows>,
) -> PyResult<DiscreteSymmetryOutput> {
    let projectors = projectors
        .map(|values| {
            values
                .into_iter()
                .map(matrix_from_rows)
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;
    let symmetry = NativeDiscreteSymmetry::new(
        projectors,
        optional_matrix(time_reversal)?,
        optional_matrix(particle_hole)?,
        optional_matrix(chiral)?,
    )
    .map_err(value_error)?;
    Ok((
        symmetry
            .projectors()
            .map(|values| values.iter().map(matrix_to_rows).collect()),
        symmetry.time_reversal().map(matrix_to_rows),
        symmetry.particle_hole().map(matrix_to_rows),
        symmetry.chiral().map(matrix_to_rows),
    ))
}

#[pyfunction(signature = (
    projectors,
    time_reversal,
    particle_hole,
    chiral,
    matrix
))]
fn discrete_symmetry_validate(
    projectors: Option<Vec<MatrixRows>>,
    time_reversal: Option<MatrixRows>,
    particle_hole: Option<MatrixRows>,
    chiral: Option<MatrixRows>,
    matrix: MatrixRows,
) -> PyResult<Vec<String>> {
    let projectors = projectors
        .map(|values| {
            values
                .into_iter()
                .map(matrix_from_rows)
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;
    let symmetry = NativeDiscreteSymmetry::new(
        projectors,
        optional_matrix(time_reversal)?,
        optional_matrix(particle_hole)?,
        optional_matrix(chiral)?,
    )
    .map_err(value_error)?;
    symmetry
        .validate(&matrix_from_rows(matrix)?)
        .map(|violations| {
            violations
                .into_iter()
                .map(|violation| violation.label().to_owned())
                .collect()
        })
        .map_err(value_error)
}

#[pyfunction]
fn lattice_gs_coefficient(vector: Vec<f64>, reference: Vec<f64>) -> PyResult<f64> {
    gram_schmidt_coefficient(&vector, &reference).map_err(value_error)
}

#[pyfunction]
fn lattice_gram_schmidt(basis: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    gram_schmidt(&basis).map_err(value_error)
}

#[pyfunction]
fn lattice_is_c_reduced(basis: Vec<Vec<f64>>, reduction_parameter: f64) -> PyResult<bool> {
    is_c_reduced(&basis, reduction_parameter).map_err(value_error)
}

#[pyfunction(signature = (basis, reduction_parameter=1.34))]
fn lattice_lll(basis: Vec<Vec<f64>>, reduction_parameter: f64) -> PyResult<LatticeReductionOutput> {
    lll_reduce(&basis, reduction_parameter)
        .map(|reduced| {
            let transformation = (0..reduced.transformation().len())
                .map(|row| {
                    (0..reduced.transformation().len())
                        .map(|column| reduced.transformation()[column][row])
                        .collect()
                })
                .collect();
            (reduced.vectors().to_vec(), transformation)
        })
        .map_err(value_error)
}

#[pyfunction(signature = (
    target,
    basis,
    neighbor_count=1,
    group_by_length=false,
    relative_tolerance=1.0e-9
))]
fn lattice_cvp(
    target: Vec<f64>,
    basis: Vec<Vec<f64>>,
    neighbor_count: usize,
    group_by_length: bool,
    relative_tolerance: f64,
) -> PyResult<Vec<Vec<i64>>> {
    closest_lattice_vectors(
        &target,
        &basis,
        neighbor_count,
        group_by_length,
        relative_tolerance,
    )
    .map_err(value_error)
}

#[pyfunction(signature = (
    basis,
    reduced=false,
    relative_tolerance=1.0e-9
))]
fn lattice_voronoi(
    basis: Vec<Vec<f64>>,
    reduced: bool,
    relative_tolerance: f64,
) -> PyResult<Vec<Vec<i64>>> {
    voronoi_neighbors(&basis, reduced, relative_tolerance).map_err(value_error)
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hamiltonian, module)?)?;
    module.add_function(wrap_pyfunction!(eigensystem, module)?)?;
    module.add_function(wrap_pyfunction!(remove_model_orbitals, module)?)?;
    module.add_function(wrap_pyfunction!(change_model_nonperiodic_vector, module)?)?;
    module.add_function(wrap_pyfunction!(make_model_supercell, module)?)?;
    module.add_function(wrap_pyfunction!(uniform_grid_chern, module)?)?;
    module.add_function(wrap_pyfunction!(second_chern_kubo, module)?)?;
    module.add_function(wrap_pyfunction!(momentum_derivatives, module)?)?;
    module.add_function(wrap_pyfunction!(finite_difference, module)?)?;
    module.add_function(wrap_pyfunction!(open_system_transmissions, module)?)?;
    module.add_function(wrap_pyfunction!(open_system_solution, module)?)?;
    module.add_function(wrap_pyfunction!(wilson_phase, module)?)?;
    module.add_function(wrap_pyfunction!(berry_flux, module)?)?;
    module.add_function(wrap_pyfunction!(transport_link, module)?)?;
    module.add_function(wrap_pyfunction!(wilson_eigenphases, module)?)?;
    module.add_function(wrap_pyfunction!(link_connection, module)?)?;
    module.add_function(wrap_pyfunction!(diagonal_observable_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(matrix_eigensystem, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_decompose, module)?)?;
    module.add_function(wrap_pyfunction!(reciprocal_path, module)?)?;
    module.add_function(wrap_pyfunction!(rmt_gaussian, module)?)?;
    module.add_function(wrap_pyfunction!(rmt_circular, module)?)?;
    module.add_function(wrap_pyfunction!(discrete_symmetry_normalize, module)?)?;
    module.add_function(wrap_pyfunction!(discrete_symmetry_validate, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_gs_coefficient, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_gram_schmidt, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_is_c_reduced, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_lll, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_cvp, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_voronoi, module)?)?;
    Ok(())
}
