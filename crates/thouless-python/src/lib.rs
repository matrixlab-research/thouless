use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use thouless::differentiation::{finite_difference_uniform, DifferenceScheme};
use thouless::model::{ModelBuilder, OrbitalId, TightBindingModel};
use thouless::{Complex64, ComplexMatrix};

type HoppingInput = (usize, usize, Vec<i32>, Vec<Vec<Complex64>>);

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
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

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hamiltonian, module)?)?;
    module.add_function(wrap_pyfunction!(eigensystem, module)?)?;
    module.add_function(wrap_pyfunction!(momentum_derivatives, module)?)?;
    module.add_function(wrap_pyfunction!(finite_difference, module)?)?;
    Ok(())
}
