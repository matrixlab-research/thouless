use pyo3::create_exception;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thouless::bands::PeriodicBands;
use thouless::continuum::{
    finite_difference_stencil, landau_ladder_coefficient, DifferentialFactor,
};
use thouless::decomposition::{
    complexify_generalized_schur, complexify_schur, eigenvectors_from_generalized_schur,
    eigenvectors_from_schur, generalized_schur, reorder_generalized_schur, reorder_schur, schur,
};
use thouless::differentiation::{finite_difference_uniform, DifferenceScheme};
use thouless::digest::{gaussian as digest_gaussian_value, uniform_pair};
use thouless::gauge::{
    axial_line_phase_quadrature, fundamental_cycles, graph_is_connected, integrate_surface_samples,
    interface_constraints_are_acyclic, line_phase_quadrature, minimum_cycle_basis,
    peierls_phases_from_fluxes, phase_from_flux, surface_quadrature,
    uniform_axial_field_peierls_phase, uniform_field_peierls_phase,
};
use thouless::geometry::ReciprocalPath;
use thouless::graph::{
    CompressedGraph, CompressionOptions, DirectedEdge, DirectedGraphBuilder, GraphError, NodeId,
};
use thouless::interpolation::{
    interpolate_current, interpolate_density, RegularField, SmoothingOptions,
};
use thouless::kpm::{
    apply_kernel, apply_operator_to_chebyshev, chebyshev_nodes, chebyshev_vectors,
    correlation_integral_factor, correlation_moments, correlation_response,
    evaluate as kpm_evaluate_native, fermi_distribution as kpm_fermi_distribution_native,
    integrate as kpm_integrate_native, kernel_weights, reconstruct as kpm_reconstruct_native,
    reconstruct_stabilized, rescale_hamiltonian, scalar_moments, velocity_operator, Kernel,
    SpectralScale,
};
use thouless::lattice_reduction::{
    closest_lattice_vectors, gram_schmidt, gram_schmidt_coefficient, is_c_reduced, lll_reduce,
    voronoi_neighbors,
};
use thouless::lead_modes::{
    propagating_modes, propagating_modes_in_declared_symmetric_subspaces,
    propagating_modes_in_subspaces, propagating_modes_with_symmetries,
    reexpress_modes_in_singular_hopping_basis_from_svd, setup_lead_linear_system,
    setup_lead_linear_system_from_svd, LeadLinearSystem, LeadLinearSystemOptions,
};
use thouless::model::{ModelBuilder, OrbitalId, TightBindingModel};
use thouless::observables::{pauli_coefficients, project_diagonal_observable};
use thouless::periodic::{fold_terms, PeriodicTerm};
use thouless::random_matrix::{circular_from_components, gaussian_from_components, SymmetryClass};
use thouless::spectrum::hermitian_eigensystem;
use thouless::symmetry::{
    particle_hole_symmetric_basis, DiscreteSymmetry as NativeDiscreteSymmetry,
};
use thouless::topology::{
    chern_numbers_on_uniform_grid, connection_from_link, local_chern_marker_from_hamiltonian,
    parallel_transport_link, plaquette_flux, quantum_geometric_tensor_from_hamiltonian_derivatives,
    second_chern_from_hamiltonian_derivatives, wilson_line_phase, wilson_loop_eigenphases,
};
use thouless::transform::{change_nonperiodic_vector, make_supercell, remove_orbitals};
use thouless::transport::{
    open_system_self_energies, partition_shot_noise, regularize_retarded_self_energy,
    retarded_lead_self_energy, solve_open_system, square_lattice_self_energy, LeadContact,
    SurfaceGreenOptions,
};
use thouless::{Complex64, ComplexMatrix};

create_exception!(thouless_python, NodeDoesNotExistError, PyIndexError);
create_exception!(thouless_python, EdgeDoesNotExistError, PyIndexError);
create_exception!(thouless_python, DisabledFeatureError, PyRuntimeError);

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
type QuantumGeometricOutput = Vec<Vec<Vec<Vec<Vec<Complex64>>>>>;
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
type ComplexTensor3 = Vec<Vec<Vec<Complex64>>>;
type KpmReconstructionOutput = (Vec<f64>, ComplexTensor3, ComplexTensor3, ComplexTensor3);
type PeriodicTermInput = (MatrixRows, Vec<i64>, bool);
type LatticeReductionOutput = (Vec<Vec<f64>>, Vec<Vec<i64>>);
type GaugeQuadratureOutput = (Vec<Vec<f64>>, Vec<Vec<f64>>);
type RegularFieldOutput = (Vec<f64>, Vec<usize>, usize, Vec<(f64, f64)>);
type ContinuumStencilInput = (Option<usize>, usize, usize);
type ContinuumCoefficientOutput = (usize, Vec<(i32, u32)>);
type ContinuumStencilOutput = (
    Vec<i32>,
    Complex64,
    Vec<u32>,
    Vec<ContinuumCoefficientOutput>,
);
type SchurOutput = (MatrixRows, MatrixRows, Vec<Complex64>);
type GeneralizedSchurOutput = (
    MatrixRows,
    MatrixRows,
    MatrixRows,
    MatrixRows,
    Vec<Complex64>,
    Vec<Complex64>,
);
type EigenvectorOutput = (Option<MatrixRows>, Option<MatrixRows>);
type BandOutput = (
    Vec<f64>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<MatrixRows>,
);
type LeadModeOutput = (
    MatrixRows,
    Vec<f64>,
    Vec<f64>,
    usize,
    MatrixRows,
    MatrixRows,
    MatrixRows,
);
type ProjectedLeadModeOutput = (
    MatrixRows,
    Vec<f64>,
    Vec<f64>,
    usize,
    MatrixRows,
    MatrixRows,
    MatrixRows,
    Vec<usize>,
);
type DiscreteSymmetryOutput = (
    Option<Vec<MatrixRows>>,
    Option<MatrixRows>,
    Option<MatrixRows>,
    Option<MatrixRows>,
);

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn graph_error(error: GraphError) -> PyErr {
    match error {
        GraphError::NodeDoesNotExist(_) => NodeDoesNotExistError::new_err(error.to_string()),
        GraphError::EdgeDoesNotExist => EdgeDoesNotExistError::new_err(error.to_string()),
        GraphError::FeatureDisabled(_) => DisabledFeatureError::new_err(error.to_string()),
        GraphError::NegativeNodesDisabled
        | GraphError::DoublyDanglingEdge
        | GraphError::NodeCountCannotDecrease { .. }
        | GraphError::ReverseIndexRequiredForDanglingTail => {
            PyValueError::new_err(error.to_string())
        }
    }
}

fn graph_index(value: i64) -> Result<usize, GraphError> {
    usize::try_from(value).map_err(|_| GraphError::EdgeDoesNotExist)
}

#[pyclass(name = "_LeadLinearSystem")]
struct PyLeadLinearSystem {
    inner: LeadLinearSystem,
}

#[pymethods]
impl PyLeadLinearSystem {
    #[getter]
    fn left(&self) -> MatrixRows {
        matrix_to_rows(self.inner.left())
    }

    #[getter]
    fn right(&self) -> Option<MatrixRows> {
        self.inner.right().map(matrix_to_rows)
    }

    #[getter]
    fn square_root_hopping(&self) -> MatrixRows {
        matrix_to_rows(self.inner.square_root_hopping())
    }

    #[getter]
    fn uses_real_arithmetic(&self) -> bool {
        self.inner.uses_real_arithmetic()
    }

    fn extract(
        &self,
        projected: Vec<Complex64>,
        inverse_bloch_factor: Complex64,
    ) -> PyResult<Vec<Complex64>> {
        self.inner
            .extract_wave_function(&projected, inverse_bloch_factor)
            .map_err(value_error)
    }
}

#[pyclass(name = "_GraphBuilder")]
struct PyGraphBuilder {
    inner: DirectedGraphBuilder,
}

#[pymethods]
impl PyGraphBuilder {
    #[new]
    #[pyo3(signature = (allow_negative_nodes=false))]
    fn new(allow_negative_nodes: bool) -> Self {
        Self {
            inner: if allow_negative_nodes {
                DirectedGraphBuilder::allowing_dangling_nodes()
            } else {
                DirectedGraphBuilder::new()
            },
        }
    }

    #[getter]
    fn allow_negative_nodes(&self) -> bool {
        self.inner.allows_dangling_nodes()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.node_count()
    }

    #[setter]
    fn set_num_nodes(&mut self, value: i64) -> PyResult<()> {
        let value = usize::try_from(value)
            .map_err(|_| PyValueError::new_err("number of nodes cannot be negative"))?;
        self.inner.set_node_count(value).map_err(graph_error)
    }

    fn reserve(&mut self, capacity: usize) {
        self.inner
            .reserve_edges(capacity.saturating_sub(self.inner.edge_count()));
    }

    fn add_edge(&mut self, tail: NodeId, head: NodeId) -> PyResult<usize> {
        self.inner
            .add_edge(DirectedEdge::new(tail, head))
            .map_err(graph_error)
    }

    fn add_edges(&mut self, edges: Vec<(NodeId, NodeId)>) -> PyResult<usize> {
        self.inner
            .extend_edges(
                edges
                    .into_iter()
                    .map(|(tail, head)| DirectedEdge::new(tail, head)),
            )
            .map_err(graph_error)
    }

    #[pyo3(signature = (
        twoway=false,
        edge_nr_translation=false,
        allow_lost_edges=false
    ))]
    fn compressed(
        &self,
        twoway: bool,
        edge_nr_translation: bool,
        allow_lost_edges: bool,
    ) -> PyResult<PyCompressedGraph> {
        self.inner
            .compress(CompressionOptions {
                reverse_index: twoway,
                edge_number_map: edge_nr_translation,
                allow_discarded_edges: allow_lost_edges,
            })
            .map(|inner| PyCompressedGraph { inner })
            .map_err(graph_error)
    }

    fn edges(&self) -> Vec<(NodeId, NodeId)> {
        self.inner
            .edges()
            .iter()
            .map(|edge| (edge.tail(), edge.head()))
            .collect()
    }

    fn dot(&self) -> String {
        let mut result = String::from("digraph g {\n");
        for edge in self.inner.edges() {
            result.push_str(&format!("  {} -> {};\n", edge.tail(), edge.head()));
        }
        result.push_str("}\n");
        result
    }
}

#[pyclass(name = "_CompressedGraph")]
struct PyCompressedGraph {
    inner: CompressedGraph,
}

#[pymethods]
impl PyCompressedGraph {
    #[getter]
    fn twoway(&self) -> bool {
        self.inner.has_reverse_index()
    }

    #[getter]
    fn edge_nr_translation(&self) -> bool {
        self.inner.has_edge_number_map()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.node_count()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.edge_count()
    }

    #[getter]
    fn num_px_edges(&self) -> usize {
        self.inner.outgoing_edge_count()
    }

    #[getter]
    fn num_xp_edges(&self) -> usize {
        self.inner.incoming_edge_count()
    }

    fn has_dangling_edges(&self) -> bool {
        self.inner.has_dangling_edges()
    }

    fn out_neighbors(&self, node: NodeId) -> PyResult<Vec<NodeId>> {
        self.inner
            .outgoing_neighbors(node)
            .map(<[NodeId]>::to_vec)
            .map_err(graph_error)
    }

    fn out_edge_ids(&self, node: NodeId) -> PyResult<Vec<usize>> {
        self.inner
            .outgoing_edge_ids(node)
            .map(|edge_ids| edge_ids.collect())
            .map_err(graph_error)
    }

    fn in_neighbors(&self, node: NodeId) -> PyResult<Vec<NodeId>> {
        self.inner
            .incoming_neighbors(node)
            .map(<[NodeId]>::to_vec)
            .map_err(graph_error)
    }

    fn in_edge_ids(&self, node: NodeId) -> PyResult<Vec<usize>> {
        self.inner
            .incoming_edge_ids(node)
            .map(<[usize]>::to_vec)
            .map_err(graph_error)
    }

    fn has_edge(&self, tail: NodeId, head: NodeId) -> PyResult<bool> {
        self.inner.contains_edge(tail, head).map_err(graph_error)
    }

    fn edge_id(&self, edge_number: i64) -> PyResult<usize> {
        self.inner
            .edge_id_from_number(graph_index(edge_number).map_err(graph_error)?)
            .map_err(graph_error)
    }

    fn first_edge_id(&self, tail: NodeId, head: NodeId) -> PyResult<usize> {
        self.inner.first_edge_id(tail, head).map_err(graph_error)
    }

    fn all_edge_ids(&self, tail: NodeId, head: NodeId) -> PyResult<Vec<usize>> {
        self.inner.all_edge_ids(tail, head).map_err(graph_error)
    }

    fn tail(&self, edge_id: i64) -> PyResult<Option<NodeId>> {
        self.inner
            .tail(graph_index(edge_id).map_err(graph_error)?)
            .map_err(graph_error)
    }

    fn head(&self, edge_id: i64) -> PyResult<NodeId> {
        self.inner
            .head(graph_index(edge_id).map_err(graph_error)?)
            .map_err(graph_error)
    }

    fn edges(&self) -> Vec<(NodeId, NodeId)> {
        self.inner
            .edges()
            .map(|edge| (edge.tail(), edge.head()))
            .collect()
    }

    fn dot(&self) -> String {
        let mut result = String::from("digraph g {\n");
        for edge in self.inner.edges() {
            result.push_str(&format!("  {} -> {};\n", edge.tail(), edge.head()));
        }
        result.push_str("}\n");
        result
    }
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
    if matrix.columns() == 0 {
        return vec![Vec::new(); matrix.rows()];
    }
    matrix
        .as_slice()
        .chunks(matrix.columns())
        .map(<[Complex64]>::to_vec)
        .collect()
}

type ScipyZgesdd = unsafe extern "C" fn(
    *mut std::ffi::c_char,
    *mut std::ffi::c_int,
    *mut std::ffi::c_int,
    *mut Complex64,
    *mut std::ffi::c_int,
    *mut f64,
    *mut Complex64,
    *mut std::ffi::c_int,
    *mut Complex64,
    *mut std::ffi::c_int,
    *mut Complex64,
    *mut std::ffi::c_int,
    *mut f64,
    *mut std::ffi::c_int,
    *mut std::ffi::c_int,
);

fn scipy_complex_svd(
    py: Python<'_>,
    matrix: &ComplexMatrix,
) -> PyResult<(ComplexMatrix, Vec<f64>, ComplexMatrix)> {
    let dimension = matrix.rows();
    if matrix.columns() != dimension {
        return Err(PyValueError::new_err("SVD input must be square"));
    }
    if dimension == 0 {
        return Ok((
            ComplexMatrix::zeros(0, 0),
            Vec::new(),
            ComplexMatrix::zeros(0, 0),
        ));
    }
    let lapack_dimension = std::ffi::c_int::try_from(dimension)
        .map_err(|_| PyValueError::new_err("SVD dimension is too large"))?;
    let lapack = py.import("scipy.linalg.cython_lapack")?;
    let capsule = lapack.getattr("__pyx_capi__")?.get_item("zgesdd")?;
    // SAFETY: SciPy publishes this function pointer through its Cython
    // LAPACK capsule. Passing the capsule's own name back to CPython verifies
    // that the requested pointer has the declared zgesdd ABI.
    let function = unsafe {
        let name = pyo3::ffi::PyCapsule_GetName(capsule.as_ptr());
        if name.is_null() {
            return Err(PyErr::fetch(py));
        }
        let pointer = pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), name);
        if pointer.is_null() {
            return Err(PyErr::fetch(py));
        }
        std::mem::transmute::<*mut std::ffi::c_void, ScipyZgesdd>(pointer)
    };

    let entry_count = dimension
        .checked_mul(dimension)
        .ok_or_else(|| PyValueError::new_err("SVD dimension is too large"))?;
    let mut column_major = vec![Complex64::new(0.0, 0.0); entry_count];
    let mut singular_values = vec![0.0; dimension];
    let mut left_vectors = vec![Complex64::new(0.0, 0.0); entry_count];
    let mut right_vectors_adjoint = vec![Complex64::new(0.0, 0.0); entry_count];
    let mut work_query = [Complex64::new(0.0, 0.0)];
    let real_work_length = 5usize
        .checked_mul(entry_count)
        .and_then(|value| value.checked_add(7 * dimension))
        .ok_or_else(|| PyValueError::new_err("SVD dimension is too large"))?;
    let mut real_work = vec![0.0; real_work_length.max(1)];
    let mut integer_work = vec![
        0;
        8usize.checked_mul(dimension).ok_or_else(|| {
            PyValueError::new_err("SVD dimension is too large")
        })?
    ];
    let mut job = b'A' as std::ffi::c_char;
    let mut rows = lapack_dimension;
    let mut columns = lapack_dimension;
    let mut leading_dimension = lapack_dimension;
    let mut left_leading_dimension = lapack_dimension;
    let mut right_leading_dimension = lapack_dimension;
    let mut work_length = -1;
    let mut info = 0;
    // SAFETY: every buffer has the documented square zgesdd size, and the
    // first call is a workspace query.
    unsafe {
        function(
            &mut job,
            &mut rows,
            &mut columns,
            column_major.as_mut_ptr(),
            &mut leading_dimension,
            singular_values.as_mut_ptr(),
            left_vectors.as_mut_ptr(),
            &mut left_leading_dimension,
            right_vectors_adjoint.as_mut_ptr(),
            &mut right_leading_dimension,
            work_query.as_mut_ptr(),
            &mut work_length,
            real_work.as_mut_ptr(),
            integer_work.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 || !work_query[0].re.is_finite() || work_query[0].re < 1.0 {
        return Err(PyRuntimeError::new_err(format!(
            "SciPy zgesdd workspace query failed with info={info}"
        )));
    }
    let queried_work_length = work_query[0].re as usize;
    work_length = std::ffi::c_int::try_from(queried_work_length)
        .map_err(|_| PyValueError::new_err("SVD workspace is too large"))?;
    let mut work = vec![Complex64::new(0.0, 0.0); queried_work_length];
    for column in 0..dimension {
        for row in 0..dimension {
            column_major[row + column * dimension] = matrix.as_slice()[row * dimension + column];
        }
    }
    // SAFETY: this is the same checked capsule and buffer layout as the
    // workspace query, now with the queried complex workspace.
    unsafe {
        function(
            &mut job,
            &mut rows,
            &mut columns,
            column_major.as_mut_ptr(),
            &mut leading_dimension,
            singular_values.as_mut_ptr(),
            left_vectors.as_mut_ptr(),
            &mut left_leading_dimension,
            right_vectors_adjoint.as_mut_ptr(),
            &mut right_leading_dimension,
            work.as_mut_ptr(),
            &mut work_length,
            real_work.as_mut_ptr(),
            integer_work.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "SciPy zgesdd failed with info={info}"
        )));
    }
    let row_major = |column_major: Vec<Complex64>| {
        (0..dimension)
            .flat_map(|row| {
                let column_major = &column_major;
                (0..dimension).map(move |column| column_major[row + column * dimension])
            })
            .collect()
    };
    Ok((
        ComplexMatrix::new(dimension, dimension, row_major(left_vectors)).map_err(value_error)?,
        singular_values,
        ComplexMatrix::new(dimension, dimension, row_major(right_vectors_adjoint))
            .map_err(value_error)?,
    ))
}

fn schur_output(decomposition: thouless::decomposition::SchurDecomposition) -> SchurOutput {
    (
        matrix_to_rows(decomposition.form()),
        matrix_to_rows(decomposition.vectors()),
        decomposition.eigenvalues().to_vec(),
    )
}

fn generalized_schur_output(
    decomposition: thouless::decomposition::GeneralizedSchurDecomposition,
) -> GeneralizedSchurOutput {
    (
        matrix_to_rows(decomposition.left_form()),
        matrix_to_rows(decomposition.right_form()),
        matrix_to_rows(decomposition.left_vectors()),
        matrix_to_rows(decomposition.right_vectors()),
        decomposition.alpha().to_vec(),
        decomposition.beta().to_vec(),
    )
}

fn eigenvector_output(vectors: thouless::decomposition::EigenvectorSet) -> EigenvectorOutput {
    (
        vectors.left().map(matrix_to_rows),
        vectors.right().map(matrix_to_rows),
    )
}

#[pyfunction]
fn validate_periodic_bands(
    cell_hamiltonian: MatrixRows,
    inter_cell_hopping: MatrixRows,
) -> PyResult<()> {
    PeriodicBands::new(
        matrix_from_rows(cell_hamiltonian)?,
        matrix_from_rows(inter_cell_hopping)?,
    )
    .map(|_| ())
    .map_err(value_error)
}

#[pyfunction]
fn lead_band_evaluation(
    cell_hamiltonian: MatrixRows,
    inter_cell_hopping: MatrixRows,
    momentum: f64,
    derivative_order: usize,
    return_eigenvectors: bool,
) -> PyResult<BandOutput> {
    let bands = PeriodicBands::new(
        matrix_from_rows(cell_hamiltonian)?,
        matrix_from_rows(inter_cell_hopping)?,
    )
    .map_err(value_error)?;
    let result = bands
        .evaluate(momentum, derivative_order, return_eigenvectors)
        .map_err(value_error)?;
    Ok((
        result.energies().to_vec(),
        result.first_derivatives().map(<[f64]>::to_vec),
        result.second_derivatives().map(<[f64]>::to_vec),
        result.eigenvectors().map(matrix_to_rows),
    ))
}

#[pyfunction]
fn reflection_shot_noise(reflection_amplitudes: MatrixRows) -> PyResult<f64> {
    partition_shot_noise(&matrix_from_rows(reflection_amplitudes)?).map_err(value_error)
}

#[pyfunction]
fn digest_uniform_pair(input: Vec<u8>, salt: Vec<u8>) -> (f64, f64) {
    uniform_pair(&input, &salt)
}

#[pyfunction]
fn digest_gaussian(input: Vec<u8>, salt: Vec<u8>) -> f64 {
    digest_gaussian_value(&input, &salt)
}

fn kpm_kernel(name: &str, strength: Option<f64>) -> PyResult<Kernel> {
    match name {
        "jackson" => Ok(Kernel::Jackson),
        "lorentz" => Ok(Kernel::Lorentz(strength.unwrap_or(4.0))),
        "none" => Ok(Kernel::None),
        _ => Err(PyValueError::new_err(
            "KPM kernel must be 'jackson', 'lorentz', or 'none'",
        )),
    }
}

#[pyfunction(signature = (hamiltonian, strict_margin=0.05, bounds=None))]
fn kpm_rescale_hamiltonian(
    hamiltonian: MatrixRows,
    strict_margin: f64,
    bounds: Option<(f64, f64)>,
) -> PyResult<(MatrixRows, f64, f64)> {
    let rescaled = rescale_hamiltonian(&matrix_from_rows(hamiltonian)?, strict_margin, bounds)
        .map_err(value_error)?;
    Ok((
        matrix_to_rows(rescaled.matrix()),
        rescaled.scale().half_width(),
        rescaled.scale().center(),
    ))
}

#[pyfunction]
fn kpm_chebyshev_vectors(
    rescaled_hamiltonian: MatrixRows,
    initial_vectors: Vec<Vec<Complex64>>,
    moment_count: usize,
) -> PyResult<ComplexTensor3> {
    chebyshev_vectors(
        &matrix_from_rows(rescaled_hamiltonian)?,
        &initial_vectors,
        moment_count,
    )
    .map_err(value_error)
}

#[pyfunction(signature = (initial_vectors, chebyshev, operator=None))]
fn kpm_scalar_moments(
    initial_vectors: Vec<Vec<Complex64>>,
    chebyshev: ComplexTensor3,
    operator: Option<MatrixRows>,
) -> PyResult<ComplexTensor3> {
    let operator = operator.map(matrix_from_rows).transpose()?;
    scalar_moments(&initial_vectors, &chebyshev, operator.as_ref()).map_err(value_error)
}

#[pyfunction]
fn kpm_apply_operator(operator: MatrixRows, chebyshev: ComplexTensor3) -> PyResult<ComplexTensor3> {
    apply_operator_to_chebyshev(&matrix_from_rows(operator)?, &chebyshev).map_err(value_error)
}

#[pyfunction(signature = (
    raw_moments,
    half_width,
    center,
    kernel="jackson",
    kernel_strength=None,
    mean=true
))]
fn kpm_reconstruct(
    raw_moments: ComplexTensor3,
    half_width: f64,
    center: f64,
    kernel: &str,
    kernel_strength: Option<f64>,
    mean: bool,
) -> PyResult<KpmReconstructionOutput> {
    let reconstruction = kpm_reconstruct_native(
        &raw_moments,
        SpectralScale::new(half_width, center).map_err(value_error)?,
        kpm_kernel(kernel, kernel_strength)?,
        mean,
    )
    .map_err(value_error)?;
    Ok((
        reconstruction.energies().to_vec(),
        reconstruction.densities().to_vec(),
        reconstruction.gammas().to_vec(),
        reconstruction.moments().to_vec(),
    ))
}

#[pyfunction]
fn kpm_reconstruct_stabilized(
    moments: ComplexTensor3,
    half_width: f64,
    center: f64,
) -> PyResult<KpmReconstructionOutput> {
    let reconstruction = reconstruct_stabilized(
        &moments,
        SpectralScale::new(half_width, center).map_err(value_error)?,
    )
    .map_err(value_error)?;
    Ok((
        reconstruction.energies().to_vec(),
        reconstruction.densities().to_vec(),
        reconstruction.gammas().to_vec(),
        reconstruction.moments().to_vec(),
    ))
}

#[pyfunction]
fn kpm_evaluate(
    stabilized_moments: ComplexTensor3,
    half_width: f64,
    center: f64,
    energies: Vec<f64>,
) -> PyResult<ComplexTensor3> {
    kpm_evaluate_native(
        &stabilized_moments,
        SpectralScale::new(half_width, center).map_err(value_error)?,
        &energies,
    )
    .map_err(value_error)
}

#[pyfunction]
fn kpm_integrate(
    gammas: ComplexTensor3,
    distribution: Vec<f64>,
    half_width: f64,
    center: f64,
) -> PyResult<Vec<Vec<Complex64>>> {
    kpm_integrate_native(
        &gammas,
        &distribution,
        SpectralScale::new(half_width, center).map_err(value_error)?,
    )
    .map_err(value_error)
}

#[pyfunction]
fn kpm_correlation_moments(
    left: ComplexTensor3,
    right: ComplexTensor3,
    mean: bool,
) -> PyResult<ComplexTensor3> {
    correlation_moments(&left, &right, mean).map_err(value_error)
}

#[pyfunction(signature = (
    moments,
    moment_count,
    kernel="jackson",
    kernel_strength=None
))]
fn kpm_correlation_integral_factor(
    moments: ComplexTensor3,
    moment_count: usize,
    kernel: &str,
    kernel_strength: Option<f64>,
) -> PyResult<Vec<Vec<Complex64>>> {
    correlation_integral_factor(&moments, moment_count, kpm_kernel(kernel, kernel_strength)?)
        .map_err(value_error)
}

#[pyfunction]
fn kpm_correlation_response(
    integral_factor: Vec<Vec<Complex64>>,
    half_width: f64,
    center: f64,
    chemical_potential: f64,
    temperature: f64,
) -> PyResult<Vec<Complex64>> {
    correlation_response(
        &integral_factor,
        SpectralScale::new(half_width, center).map_err(value_error)?,
        chemical_potential,
        temperature,
    )
    .map_err(value_error)
}

#[pyfunction(signature = (moment_count, kernel="jackson", kernel_strength=None))]
fn kpm_kernel_weights(
    moment_count: usize,
    kernel: &str,
    kernel_strength: Option<f64>,
) -> PyResult<Vec<f64>> {
    kernel_weights(moment_count, kpm_kernel(kernel, kernel_strength)?).map_err(value_error)
}

#[pyfunction(signature = (moments, kernel="jackson", kernel_strength=None))]
fn kpm_apply_kernel(
    moments: Vec<Vec<Complex64>>,
    kernel: &str,
    kernel_strength: Option<f64>,
) -> PyResult<Vec<Vec<Complex64>>> {
    apply_kernel(&moments, kpm_kernel(kernel, kernel_strength)?).map_err(value_error)
}

#[pyfunction]
fn kpm_chebyshev_nodes(sample_count: usize) -> PyResult<Vec<f64>> {
    chebyshev_nodes(sample_count).map_err(value_error)
}

#[pyfunction]
fn kpm_fermi_distribution(
    energies: Vec<f64>,
    chemical_potential: f64,
    temperature: f64,
) -> PyResult<Vec<f64>> {
    kpm_fermi_distribution_native(&energies, chemical_potential, temperature).map_err(value_error)
}

#[pyfunction]
fn kpm_velocity_operator(
    hamiltonian: MatrixRows,
    positions: Vec<Vec<f64>>,
    direction: usize,
) -> PyResult<MatrixRows> {
    velocity_operator(&matrix_from_rows(hamiltonian)?, &positions, direction)
        .map(|matrix| matrix_to_rows(&matrix))
        .map_err(value_error)
}

#[pyfunction]
fn periodic_fold_terms(terms: Vec<PeriodicTermInput>, momentum: Vec<f64>) -> PyResult<MatrixRows> {
    let terms = terms
        .into_iter()
        .map(|(value, translation, include_adjoint)| {
            matrix_from_rows(value)
                .map(|value| PeriodicTerm::new(value, translation, include_adjoint))
        })
        .collect::<PyResult<Vec<_>>>()?;
    fold_terms(&terms, &momentum)
        .map(|matrix| matrix_to_rows(&matrix))
        .map_err(value_error)
}

#[pyfunction]
fn lead_propagating_modes(
    cell_hamiltonian: MatrixRows,
    inter_cell_hopping: MatrixRows,
) -> PyResult<LeadModeOutput> {
    propagating_modes(
        &matrix_from_rows(cell_hamiltonian)?,
        &matrix_from_rows(inter_cell_hopping)?,
    )
    .map(|modes| {
        (
            matrix_to_rows(modes.wave_functions()),
            modes.velocities().to_vec(),
            modes.momenta().to_vec(),
            modes.incoming_count(),
            matrix_to_rows(modes.stabilized_vectors()),
            matrix_to_rows(modes.stabilized_vectors_lambda_inverse()),
            matrix_to_rows(modes.square_root_hopping()),
        )
    })
    .map_err(value_error)
}

#[pyfunction(signature = (
    cell_hamiltonian,
    inter_cell_hopping,
    tolerance_multiplier=1.0e6,
    singular_hopping_basis=false,
    force_imaginary_stabilizer=false,
    regularize_pencil=None
))]
fn lead_setup_linear_system(
    py: Python<'_>,
    cell_hamiltonian: MatrixRows,
    inter_cell_hopping: MatrixRows,
    tolerance_multiplier: f64,
    singular_hopping_basis: bool,
    force_imaginary_stabilizer: bool,
    regularize_pencil: Option<bool>,
) -> PyResult<PyLeadLinearSystem> {
    let options = if singular_hopping_basis {
        LeadLinearSystemOptions::singular_hopping_basis(
            tolerance_multiplier,
            force_imaginary_stabilizer,
            regularize_pencil,
        )
    } else {
        LeadLinearSystemOptions::automatic(tolerance_multiplier)
    };
    let cell_hamiltonian = matrix_from_rows(cell_hamiltonian)?;
    let inter_cell_hopping = matrix_from_rows(inter_cell_hopping)?;
    let result = if singular_hopping_basis {
        let (left_vectors, singular_values, right_vectors_adjoint) =
            scipy_complex_svd(py, &inter_cell_hopping)?;
        setup_lead_linear_system_from_svd(
            &cell_hamiltonian,
            &inter_cell_hopping,
            &left_vectors,
            &singular_values,
            &right_vectors_adjoint,
            options,
        )
    } else {
        setup_lead_linear_system(&cell_hamiltonian, &inter_cell_hopping, options)
    };
    result
        .map(|inner| PyLeadLinearSystem { inner })
        .map_err(value_error)
}

#[pyfunction]
fn lead_projected_modes(
    cell_hamiltonian: MatrixRows,
    inter_cell_hopping: MatrixRows,
    projectors: Vec<MatrixRows>,
) -> PyResult<ProjectedLeadModeOutput> {
    let projectors = projectors
        .into_iter()
        .map(matrix_from_rows)
        .collect::<PyResult<Vec<_>>>()?;
    propagating_modes_in_subspaces(
        &matrix_from_rows(cell_hamiltonian)?,
        &matrix_from_rows(inter_cell_hopping)?,
        &projectors,
    )
    .map(|projected| {
        let modes = projected.modes();
        (
            matrix_to_rows(modes.wave_functions()),
            modes.velocities().to_vec(),
            modes.momenta().to_vec(),
            modes.incoming_count(),
            matrix_to_rows(modes.stabilized_vectors()),
            matrix_to_rows(modes.stabilized_vectors_lambda_inverse()),
            matrix_to_rows(modes.square_root_hopping()),
            projected.block_incoming_counts().to_vec(),
        )
    })
    .map_err(value_error)
}

#[pyfunction(signature = (
    cell_hamiltonian,
    inter_cell_hopping,
    time_reversal=None,
    particle_hole=None,
    chiral=None
))]
fn lead_symmetric_modes(
    cell_hamiltonian: MatrixRows,
    inter_cell_hopping: MatrixRows,
    time_reversal: Option<MatrixRows>,
    particle_hole: Option<MatrixRows>,
    chiral: Option<MatrixRows>,
) -> PyResult<LeadModeOutput> {
    let time_reversal = optional_matrix(time_reversal)?;
    let particle_hole = optional_matrix(particle_hole)?;
    let chiral = optional_matrix(chiral)?;
    propagating_modes_with_symmetries(
        &matrix_from_rows(cell_hamiltonian)?,
        &matrix_from_rows(inter_cell_hopping)?,
        time_reversal.as_ref(),
        particle_hole.as_ref(),
        chiral.as_ref(),
    )
    .map(|modes| {
        (
            matrix_to_rows(modes.wave_functions()),
            modes.velocities().to_vec(),
            modes.momenta().to_vec(),
            modes.incoming_count(),
            matrix_to_rows(modes.stabilized_vectors()),
            matrix_to_rows(modes.stabilized_vectors_lambda_inverse()),
            matrix_to_rows(modes.square_root_hopping()),
        )
    })
    .map_err(value_error)
}

#[pyfunction(signature = (
    cell_hamiltonian,
    inter_cell_hopping,
    relative_singular_tolerance,
    time_reversal=None,
    particle_hole=None,
    chiral=None
))]
fn lead_singular_basis_modes(
    py: Python<'_>,
    cell_hamiltonian: MatrixRows,
    inter_cell_hopping: MatrixRows,
    relative_singular_tolerance: f64,
    time_reversal: Option<MatrixRows>,
    particle_hole: Option<MatrixRows>,
    chiral: Option<MatrixRows>,
) -> PyResult<LeadModeOutput> {
    let cell_hamiltonian = matrix_from_rows(cell_hamiltonian)?;
    let inter_cell_hopping = matrix_from_rows(inter_cell_hopping)?;
    let time_reversal = optional_matrix(time_reversal)?;
    let particle_hole = optional_matrix(particle_hole)?;
    let chiral = optional_matrix(chiral)?;
    let modes = if time_reversal.is_some() || particle_hole.is_some() || chiral.is_some() {
        propagating_modes_with_symmetries(
            &cell_hamiltonian,
            &inter_cell_hopping,
            time_reversal.as_ref(),
            particle_hole.as_ref(),
            chiral.as_ref(),
        )
    } else {
        propagating_modes(&cell_hamiltonian, &inter_cell_hopping)
    }
    .and_then(|modes| {
        let (_, singular_values, right_vectors_adjoint) =
            scipy_complex_svd(py, &inter_cell_hopping)
                .map_err(|_| thouless::lead_modes::LeadModeError::DecompositionFailure)?;
        reexpress_modes_in_singular_hopping_basis_from_svd(
            &modes,
            &inter_cell_hopping,
            &singular_values,
            &right_vectors_adjoint,
            relative_singular_tolerance,
        )
    })
    .map_err(value_error)?;
    Ok((
        matrix_to_rows(modes.wave_functions()),
        modes.velocities().to_vec(),
        modes.momenta().to_vec(),
        modes.incoming_count(),
        matrix_to_rows(modes.stabilized_vectors()),
        matrix_to_rows(modes.stabilized_vectors_lambda_inverse()),
        matrix_to_rows(modes.square_root_hopping()),
    ))
}

#[pyfunction(signature = (
    cell_hamiltonian,
    inter_cell_hopping,
    projectors,
    time_reversal=None,
    particle_hole=None,
    chiral=None
))]
fn lead_symmetric_projected_modes(
    cell_hamiltonian: MatrixRows,
    inter_cell_hopping: MatrixRows,
    projectors: Vec<MatrixRows>,
    time_reversal: Option<MatrixRows>,
    particle_hole: Option<MatrixRows>,
    chiral: Option<MatrixRows>,
) -> PyResult<ProjectedLeadModeOutput> {
    let projectors = projectors
        .into_iter()
        .map(matrix_from_rows)
        .collect::<PyResult<Vec<_>>>()?;
    let time_reversal = optional_matrix(time_reversal)?;
    let particle_hole = optional_matrix(particle_hole)?;
    let chiral = optional_matrix(chiral)?;
    propagating_modes_in_declared_symmetric_subspaces(
        &matrix_from_rows(cell_hamiltonian)?,
        &matrix_from_rows(inter_cell_hopping)?,
        &projectors,
        time_reversal.as_ref(),
        particle_hole.as_ref(),
        chiral.as_ref(),
    )
    .map(|projected| {
        let modes = projected.modes();
        (
            matrix_to_rows(modes.wave_functions()),
            modes.velocities().to_vec(),
            modes.momenta().to_vec(),
            modes.incoming_count(),
            matrix_to_rows(modes.stabilized_vectors()),
            matrix_to_rows(modes.stabilized_vectors_lambda_inverse()),
            matrix_to_rows(modes.square_root_hopping()),
            projected.block_incoming_counts().to_vec(),
        )
    })
    .map_err(value_error)
}

#[pyfunction(signature = (
    cell_hamiltonian,
    inter_cell_hopping,
    energy=0.0,
    broadening=None,
    maximum_rank=None
))]
fn lead_retarded_self_energy(
    cell_hamiltonian: MatrixRows,
    inter_cell_hopping: MatrixRows,
    energy: f64,
    broadening: Option<f64>,
    maximum_rank: Option<usize>,
) -> PyResult<MatrixRows> {
    let mut options = SurfaceGreenOptions::default();
    if let Some(broadening) = broadening {
        options.broadening = broadening;
    }
    let self_energy = retarded_lead_self_energy(
        &matrix_from_rows(cell_hamiltonian)?,
        &matrix_from_rows(inter_cell_hopping)?,
        energy,
        options,
    )
    .and_then(|self_energy| regularize_retarded_self_energy(&self_energy, maximum_rank))
    .map_err(value_error)?;
    Ok(matrix_to_rows(&self_energy))
}

#[pyfunction]
fn square_strip_self_energy(width: usize, hopping: f64, fermi_energy: f64) -> PyResult<MatrixRows> {
    square_lattice_self_energy(width, hopping, fermi_energy)
        .map(|self_energy| matrix_to_rows(&self_energy))
        .map_err(value_error)
}

#[pyfunction]
fn dense_schur(matrix: MatrixRows) -> PyResult<SchurOutput> {
    schur(&matrix_from_rows(matrix)?)
        .map(schur_output)
        .map_err(value_error)
}

#[pyfunction]
fn dense_reorder_schur(
    form: MatrixRows,
    vectors: MatrixRows,
    selected: Vec<bool>,
) -> PyResult<SchurOutput> {
    reorder_schur(
        &matrix_from_rows(form)?,
        &matrix_from_rows(vectors)?,
        &selected,
    )
    .map(schur_output)
    .map_err(value_error)
}

#[pyfunction]
fn dense_schur_eigenvectors(
    form: MatrixRows,
    vectors: MatrixRows,
    selected: Vec<bool>,
    compute_left: bool,
    compute_right: bool,
) -> PyResult<EigenvectorOutput> {
    eigenvectors_from_schur(
        &matrix_from_rows(form)?,
        &matrix_from_rows(vectors)?,
        &selected,
        compute_left,
        compute_right,
    )
    .map(eigenvector_output)
    .map_err(value_error)
}

#[pyfunction]
fn dense_complexify_schur(form: MatrixRows, vectors: MatrixRows) -> PyResult<SchurOutput> {
    complexify_schur(&matrix_from_rows(form)?, &matrix_from_rows(vectors)?)
        .map(schur_output)
        .map_err(value_error)
}

#[pyfunction]
fn dense_generalized_schur(
    left: MatrixRows,
    right: MatrixRows,
) -> PyResult<GeneralizedSchurOutput> {
    generalized_schur(&matrix_from_rows(left)?, &matrix_from_rows(right)?)
        .map(generalized_schur_output)
        .map_err(value_error)
}

#[pyfunction]
fn dense_reorder_generalized_schur(
    left_form: MatrixRows,
    right_form: MatrixRows,
    left_vectors: MatrixRows,
    right_vectors: MatrixRows,
    selected: Vec<bool>,
) -> PyResult<GeneralizedSchurOutput> {
    reorder_generalized_schur(
        &matrix_from_rows(left_form)?,
        &matrix_from_rows(right_form)?,
        &matrix_from_rows(left_vectors)?,
        &matrix_from_rows(right_vectors)?,
        &selected,
    )
    .map(generalized_schur_output)
    .map_err(value_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn dense_generalized_schur_eigenvectors(
    left_form: MatrixRows,
    right_form: MatrixRows,
    left_vectors: MatrixRows,
    right_vectors: MatrixRows,
    selected: Vec<bool>,
    compute_left: bool,
    compute_right: bool,
) -> PyResult<EigenvectorOutput> {
    eigenvectors_from_generalized_schur(
        &matrix_from_rows(left_form)?,
        &matrix_from_rows(right_form)?,
        &matrix_from_rows(left_vectors)?,
        &matrix_from_rows(right_vectors)?,
        &selected,
        compute_left,
        compute_right,
    )
    .map(eigenvector_output)
    .map_err(value_error)
}

#[pyfunction]
fn dense_complexify_generalized_schur(
    left_form: MatrixRows,
    right_form: MatrixRows,
    left_vectors: MatrixRows,
    right_vectors: MatrixRows,
) -> PyResult<GeneralizedSchurOutput> {
    complexify_generalized_schur(
        &matrix_from_rows(left_form)?,
        &matrix_from_rows(right_form)?,
        &matrix_from_rows(left_vectors)?,
        &matrix_from_rows(right_vectors)?,
    )
    .map(generalized_schur_output)
    .map_err(value_error)
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
fn quantum_geometric_tensor_kubo(
    hamiltonians: Vec<Vec<Vec<Complex64>>>,
    derivatives: Vec<Vec<Vec<Vec<Complex64>>>>,
    occupied_states: Vec<usize>,
) -> PyResult<QuantumGeometricOutput> {
    if hamiltonians.len() != derivatives.len() {
        return Err(PyValueError::new_err(
            "Hamiltonian and derivative batch sizes must match",
        ));
    }
    hamiltonians
        .into_iter()
        .zip(derivatives)
        .map(|(hamiltonian, derivative_group)| {
            let hamiltonian = matrix_from_rows(hamiltonian)?;
            let derivative_group = derivative_group
                .into_iter()
                .map(matrix_from_rows)
                .collect::<PyResult<Vec<_>>>()?;
            let tensor = quantum_geometric_tensor_from_hamiltonian_derivatives(
                &hamiltonian,
                &derivative_group,
                &occupied_states,
            )
            .map_err(value_error)?;
            let direction_count = tensor.direction_count();
            Ok((0..direction_count)
                .map(|first| {
                    (0..direction_count)
                        .map(|second| {
                            matrix_to_rows(
                                tensor
                                    .component(first, second)
                                    .expect("validated tensor component"),
                            )
                        })
                        .collect()
                })
                .collect())
        })
        .collect()
}

#[pyfunction]
fn local_chern_marker_kubo(
    hamiltonian: Vec<Vec<Complex64>>,
    positions: Vec<Vec<f64>>,
    occupied_states: Vec<usize>,
    cell_area: f64,
) -> PyResult<Vec<f64>> {
    let hamiltonian = matrix_from_rows(hamiltonian)?;
    let positions = positions
        .into_iter()
        .map(|position| {
            if position.len() != 2 {
                return Err(PyValueError::new_err(
                    "local Chern marker positions must have two coordinates",
                ));
            }
            Ok([position[0], position[1]])
        })
        .collect::<PyResult<Vec<_>>>()?;
    local_chern_marker_from_hamiltonian(&hamiltonian, &positions, &occupied_states, cell_area)
        .map_err(value_error)
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
fn open_system_embedded_self_energies(
    device_hamiltonian: Vec<Vec<Complex64>>,
    leads: Vec<LeadInput>,
    energy: f64,
    broadening: Option<f64>,
) -> PyResult<Vec<Vec<Vec<Complex64>>>> {
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
    open_system_self_energies(&device_hamiltonian, &leads, energy, options)
        .map(|values| values.iter().map(matrix_to_rows).collect())
        .map_err(value_error)
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
fn particle_hole_basis(
    wave_functions: MatrixRows,
    particle_hole: MatrixRows,
) -> PyResult<(MatrixRows, Vec<usize>)> {
    let basis = particle_hole_symmetric_basis(
        &matrix_from_rows(wave_functions)?,
        &matrix_from_rows(particle_hole)?,
    )
    .map_err(value_error)?;
    Ok((
        matrix_to_rows(basis.wave_functions()),
        basis.ordering().to_vec(),
    ))
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

#[pyfunction]
fn gauge_surface_quadrature(loop_points: Vec<Vec<f64>>) -> PyResult<GaugeQuadratureOutput> {
    surface_quadrature(&loop_points)
        .map(|quadrature| {
            (
                quadrature.points().to_vec(),
                quadrature.oriented_weights().to_vec(),
            )
        })
        .map_err(value_error)
}

#[pyfunction]
fn gauge_line_quadrature(first: Vec<f64>, second: Vec<f64>) -> PyResult<GaugeQuadratureOutput> {
    line_phase_quadrature(&first, &second)
        .map(|quadrature| {
            (
                quadrature.points().to_vec(),
                quadrature.oriented_weights().to_vec(),
            )
        })
        .map_err(value_error)
}

#[pyfunction]
fn gauge_axial_line_quadrature(
    first: Vec<f64>,
    second: Vec<f64>,
    axis: Vec<f64>,
) -> PyResult<GaugeQuadratureOutput> {
    axial_line_phase_quadrature(&first, &second, &axis)
        .map(|quadrature| {
            (
                quadrature.points().to_vec(),
                quadrature.oriented_weights().to_vec(),
            )
        })
        .map_err(value_error)
}

#[pyfunction]
fn gauge_surface_integral(
    oriented_weights: Vec<Vec<f64>>,
    samples: Vec<Vec<f64>>,
) -> PyResult<f64> {
    integrate_surface_samples(&oriented_weights, &samples).map_err(value_error)
}

#[pyfunction]
fn gauge_uniform_field_phase(
    first: Vec<f64>,
    second: Vec<f64>,
    field: Vec<f64>,
) -> PyResult<Complex64> {
    uniform_field_peierls_phase(&first, &second, &field).map_err(value_error)
}

#[pyfunction]
fn gauge_uniform_axial_field_phase(
    first: Vec<f64>,
    second: Vec<f64>,
    axis: Vec<f64>,
    field: f64,
) -> PyResult<Complex64> {
    uniform_axial_field_peierls_phase(&first, &second, &axis, field).map_err(value_error)
}

#[pyfunction]
fn gauge_phase_from_flux(flux: f64) -> PyResult<Complex64> {
    phase_from_flux(flux).map_err(value_error)
}

#[pyfunction]
fn gauge_minimum_cycle_basis(
    node_count: usize,
    undirected_edges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<usize>>> {
    minimum_cycle_basis(node_count, &undirected_edges).map_err(value_error)
}

#[pyfunction]
fn gauge_graph_is_connected(
    node_count: usize,
    undirected_edges: Vec<(usize, usize)>,
) -> PyResult<bool> {
    graph_is_connected(node_count, &undirected_edges).map_err(value_error)
}

#[pyfunction]
fn gauge_interfaces_are_acyclic(node_count: usize, interfaces: Vec<Vec<usize>>) -> PyResult<bool> {
    interface_constraints_are_acyclic(node_count, &interfaces).map_err(value_error)
}

#[pyfunction]
fn gauge_fundamental_cycles(
    node_count: usize,
    undirected_edges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<usize>>> {
    fundamental_cycles(node_count, &undirected_edges).map_err(value_error)
}

type EdgePhase = (usize, usize, Complex64);

#[pyfunction]
fn gauge_peierls_phases(
    node_count: usize,
    undirected_edges: Vec<(usize, usize)>,
    cycle_fluxes: Vec<f64>,
) -> PyResult<Vec<EdgePhase>> {
    peierls_phases_from_fluxes(node_count, &undirected_edges, &cycle_fluxes).map_err(value_error)
}

fn regular_field_output(field: RegularField) -> RegularFieldOutput {
    (
        field.values().to_vec(),
        field.shape().to_vec(),
        field.components(),
        field.bounds().to_vec(),
    )
}

#[pyfunction(signature = (
    points,
    values,
    reference_edges,
    absolute_width=None,
    relative_width=None,
    samples_per_width=9
))]
fn interpolate_density_field(
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    reference_edges: Vec<(Vec<f64>, Vec<f64>)>,
    absolute_width: Option<f64>,
    relative_width: Option<f64>,
    samples_per_width: usize,
) -> PyResult<RegularFieldOutput> {
    interpolate_density(
        &points,
        &values,
        &reference_edges,
        SmoothingOptions {
            absolute_width,
            relative_width,
            samples_per_width,
        },
    )
    .map(regular_field_output)
    .map_err(value_error)
}

#[pyfunction(signature = (
    edges,
    currents,
    absolute_width=None,
    relative_width=None,
    samples_per_width=9
))]
fn interpolate_current_field(
    edges: Vec<(Vec<f64>, Vec<f64>)>,
    currents: Vec<f64>,
    absolute_width: Option<f64>,
    relative_width: Option<f64>,
    samples_per_width: usize,
) -> PyResult<RegularFieldOutput> {
    interpolate_current(
        &edges,
        &currents,
        SmoothingOptions {
            absolute_width,
            relative_width,
            samples_per_width,
        },
    )
    .map(regular_field_output)
    .map_err(value_error)
}

#[pyfunction]
fn continuum_finite_difference_stencil(
    dimension: usize,
    factors: Vec<ContinuumStencilInput>,
) -> PyResult<Vec<ContinuumStencilOutput>> {
    let factors = factors
        .into_iter()
        .map(|(axis, value, power)| {
            if let Some(axis) = axis {
                Ok(DifferentialFactor::Momentum { axis, power })
            } else if power == 1 {
                Ok(DifferentialFactor::Coefficient(value))
            } else {
                Err(PyValueError::new_err(
                    "coefficient factors must have unit descriptor power",
                ))
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    finite_difference_stencil(dimension, &factors)
        .map(|terms| {
            terms
                .into_iter()
                .map(|term| {
                    (
                        term.wave_offset().to_vec(),
                        term.weight(),
                        term.inverse_spacing_powers().to_vec(),
                        term.coefficients()
                            .iter()
                            .map(|coefficient| {
                                (
                                    coefficient.id(),
                                    coefficient
                                        .shifts()
                                        .iter()
                                        .map(|shift| (shift.numerator(), shift.denominator()))
                                        .collect(),
                                )
                            })
                            .collect(),
                    )
                })
                .collect()
        })
        .map_err(value_error)
}

#[pyfunction]
fn continuum_landau_ladder_coefficient(
    ladder_powers: Vec<i32>,
    initial_level: usize,
    magnetic_field: f64,
) -> PyResult<f64> {
    landau_ladder_coefficient(&ladder_powers, initial_level, magnetic_field).map_err(value_error)
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "NodeDoesNotExistError",
        module.py().get_type::<NodeDoesNotExistError>(),
    )?;
    module.add(
        "EdgeDoesNotExistError",
        module.py().get_type::<EdgeDoesNotExistError>(),
    )?;
    module.add(
        "DisabledFeatureError",
        module.py().get_type::<DisabledFeatureError>(),
    )?;
    module.add_class::<PyGraphBuilder>()?;
    module.add_class::<PyCompressedGraph>()?;
    module.add_class::<PyLeadLinearSystem>()?;
    module.add_function(wrap_pyfunction!(validate_periodic_bands, module)?)?;
    module.add_function(wrap_pyfunction!(lead_band_evaluation, module)?)?;
    module.add_function(wrap_pyfunction!(reflection_shot_noise, module)?)?;
    module.add_function(wrap_pyfunction!(digest_uniform_pair, module)?)?;
    module.add_function(wrap_pyfunction!(digest_gaussian, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_rescale_hamiltonian, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_chebyshev_vectors, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_scalar_moments, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_apply_operator, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_reconstruct, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_reconstruct_stabilized, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_integrate, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_correlation_moments, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_correlation_integral_factor, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_correlation_response, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_kernel_weights, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_apply_kernel, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_chebyshev_nodes, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_fermi_distribution, module)?)?;
    module.add_function(wrap_pyfunction!(kpm_velocity_operator, module)?)?;
    module.add_function(wrap_pyfunction!(periodic_fold_terms, module)?)?;
    module.add_function(wrap_pyfunction!(lead_propagating_modes, module)?)?;
    module.add_function(wrap_pyfunction!(lead_setup_linear_system, module)?)?;
    module.add_function(wrap_pyfunction!(lead_projected_modes, module)?)?;
    module.add_function(wrap_pyfunction!(lead_symmetric_modes, module)?)?;
    module.add_function(wrap_pyfunction!(lead_singular_basis_modes, module)?)?;
    module.add_function(wrap_pyfunction!(lead_symmetric_projected_modes, module)?)?;
    module.add_function(wrap_pyfunction!(lead_retarded_self_energy, module)?)?;
    module.add_function(wrap_pyfunction!(square_strip_self_energy, module)?)?;
    module.add_function(wrap_pyfunction!(dense_schur, module)?)?;
    module.add_function(wrap_pyfunction!(dense_reorder_schur, module)?)?;
    module.add_function(wrap_pyfunction!(dense_schur_eigenvectors, module)?)?;
    module.add_function(wrap_pyfunction!(dense_complexify_schur, module)?)?;
    module.add_function(wrap_pyfunction!(dense_generalized_schur, module)?)?;
    module.add_function(wrap_pyfunction!(dense_reorder_generalized_schur, module)?)?;
    module.add_function(wrap_pyfunction!(
        dense_generalized_schur_eigenvectors,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        dense_complexify_generalized_schur,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(hamiltonian, module)?)?;
    module.add_function(wrap_pyfunction!(eigensystem, module)?)?;
    module.add_function(wrap_pyfunction!(remove_model_orbitals, module)?)?;
    module.add_function(wrap_pyfunction!(change_model_nonperiodic_vector, module)?)?;
    module.add_function(wrap_pyfunction!(make_model_supercell, module)?)?;
    module.add_function(wrap_pyfunction!(uniform_grid_chern, module)?)?;
    module.add_function(wrap_pyfunction!(second_chern_kubo, module)?)?;
    module.add_function(wrap_pyfunction!(quantum_geometric_tensor_kubo, module)?)?;
    module.add_function(wrap_pyfunction!(local_chern_marker_kubo, module)?)?;
    module.add_function(wrap_pyfunction!(momentum_derivatives, module)?)?;
    module.add_function(wrap_pyfunction!(finite_difference, module)?)?;
    module.add_function(wrap_pyfunction!(open_system_transmissions, module)?)?;
    module.add_function(wrap_pyfunction!(
        open_system_embedded_self_energies,
        module
    )?)?;
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
    module.add_function(wrap_pyfunction!(particle_hole_basis, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_gs_coefficient, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_gram_schmidt, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_is_c_reduced, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_lll, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_cvp, module)?)?;
    module.add_function(wrap_pyfunction!(lattice_voronoi, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_surface_quadrature, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_line_quadrature, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_axial_line_quadrature, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_surface_integral, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_uniform_field_phase, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_uniform_axial_field_phase, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_phase_from_flux, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_minimum_cycle_basis, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_graph_is_connected, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_interfaces_are_acyclic, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_fundamental_cycles, module)?)?;
    module.add_function(wrap_pyfunction!(gauge_peierls_phases, module)?)?;
    module.add_function(wrap_pyfunction!(interpolate_density_field, module)?)?;
    module.add_function(wrap_pyfunction!(interpolate_current_field, module)?)?;
    module.add_function(wrap_pyfunction!(
        continuum_finite_difference_stencil,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        continuum_landau_ladder_coefficient,
        module
    )?)?;
    Ok(())
}
