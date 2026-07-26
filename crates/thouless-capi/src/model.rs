use std::str;

use thouless::model::{Lattice, ModelBuilder, OrbitalId, TightBindingModel};
use thouless::transform::{make_finite_cluster, make_finite_geometry, remove_orbitals, FiniteSite};
use thouless::Complex64;

use crate::{
    borrowed_mut_slice, borrowed_slice, boundary, read_complex_matrix, read_real_matrix,
    write_complex_matrix, AbiError, ThoulessC64MatrixMut, ThoulessC64MatrixView, ThoulessComplex64,
    ThoulessF64MatrixView, ThoulessStatus,
};

/// Opaque exclusively mutable model builder. It is not thread-safe.
pub struct ThoulessModelBuilder {
    inner: Option<ModelBuilder>,
    orbitals: Vec<OrbitalId>,
}

/// Opaque immutable model. Concurrent read-only calls are thread-safe.
pub struct ThoulessModel {
    pub(crate) inner: TightBindingModel,
}

unsafe fn builder_mut<'a>(
    handle: *mut ThoulessModelBuilder,
) -> Result<&'a mut ThoulessModelBuilder, AbiError> {
    if handle.is_null() {
        return Err(AbiError::new(
            ThoulessStatus::NullPointer,
            "model builder is null",
        ));
    }
    // SAFETY: non-null builder handles are created and uniquely owned by this ABI.
    Ok(unsafe { &mut *handle })
}

pub(crate) unsafe fn model_ref<'a>(
    handle: *const ThoulessModel,
) -> Result<&'a ThoulessModel, AbiError> {
    if handle.is_null() {
        return Err(AbiError::new(ThoulessStatus::NullPointer, "model is null"));
    }
    // SAFETY: non-null model handles are created by this ABI and immutable.
    Ok(unsafe { &*handle })
}

unsafe fn output_handle<T>(output: *mut *mut T, value: T) -> Result<(), AbiError> {
    if output.is_null() {
        return Err(AbiError::new(
            ThoulessStatus::NullPointer,
            "output handle is null",
        ));
    }
    // SAFETY: the caller supplies writable storage for one pointer.
    unsafe { *output = Box::into_raw(Box::new(value)) };
    Ok(())
}

fn builder_inner(builder: &mut ThoulessModelBuilder) -> Result<&mut ModelBuilder, AbiError> {
    builder
        .inner
        .as_mut()
        .ok_or_else(|| AbiError::invalid("the model builder has already been consumed"))
}

fn orbital(builder: &ThoulessModelBuilder, index: usize) -> Result<OrbitalId, AbiError> {
    builder
        .orbitals
        .get(index)
        .copied()
        .ok_or_else(|| AbiError::invalid(format!("unknown orbital index {index}")))
}

/// Create a model builder from Cartesian primitive-vector rows.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_builder_create(
    primitive_vectors: ThoulessF64MatrixView,
    periodic_axes: *const usize,
    periodic_axis_count: usize,
    output: *mut *mut ThoulessModelBuilder,
) -> ThoulessStatus {
    boundary(|| {
        let primitive_vectors =
            unsafe { read_real_matrix(primitive_vectors, "primitive vectors")? };
        let periodic_axes =
            unsafe { borrowed_slice(periodic_axes, periodic_axis_count, "periodic axes")? };
        let lattice = Lattice::new(
            (0..primitive_vectors.rows())
                .map(|row| {
                    primitive_vectors.as_slice()
                        [row * primitive_vectors.columns()..(row + 1) * primitive_vectors.columns()]
                        .to_vec()
                })
                .collect(),
            periodic_axes.to_vec(),
        )
        .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe {
            output_handle(
                output,
                ThoulessModelBuilder {
                    inner: Some(ModelBuilder::new(lattice)),
                    orbitals: Vec::new(),
                },
            )
        }
    })
}

/// Destroy a model builder. Passing null is a no-op.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_builder_destroy(
    handle: *mut ThoulessModelBuilder,
) -> ThoulessStatus {
    boundary(|| {
        if !handle.is_null() {
            // SAFETY: the caller transfers the one owned handle back exactly once.
            drop(unsafe { Box::from_raw(handle) });
        }
        Ok(())
    })
}

/// Add one localized orbital or multicomponent subspace.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_builder_add_orbital(
    handle: *mut ThoulessModelBuilder,
    label: *const u8,
    label_length: usize,
    reduced_position: *const f64,
    position_length: usize,
    degrees_of_freedom: usize,
    output_orbital: *mut usize,
) -> ThoulessStatus {
    boundary(|| {
        if output_orbital.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "output orbital is null",
            ));
        }
        let label = unsafe { borrowed_slice(label, label_length, "orbital label")? };
        let label = str::from_utf8(label)
            .map_err(|_| AbiError::invalid("orbital label is not valid UTF-8"))?;
        let position =
            unsafe { borrowed_slice(reduced_position, position_length, "reduced position")? };
        let builder = unsafe { builder_mut(handle)? };
        let identifier = builder_inner(builder)?
            .add_orbital_with_dof(label, position.iter().copied(), degrees_of_freedom)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        builder.orbitals.push(identifier);
        // SAFETY: output_orbital was checked and points to one writable usize.
        unsafe { *output_orbital = builder.orbitals.len() - 1 };
        Ok(())
    })
}

/// Set a real scalar onsite energy.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_builder_set_onsite(
    handle: *mut ThoulessModelBuilder,
    orbital_index: usize,
    energy: f64,
) -> ThoulessStatus {
    boundary(|| {
        let builder = unsafe { builder_mut(handle)? };
        let orbital = orbital(builder, orbital_index)?;
        builder_inner(builder)?
            .set_onsite(orbital, energy)
            .map_err(|error| AbiError::invalid(error.to_string()))
    })
}

/// Set a Hermitian onsite block.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_builder_set_onsite_block(
    handle: *mut ThoulessModelBuilder,
    orbital_index: usize,
    block: ThoulessC64MatrixView,
) -> ThoulessStatus {
    boundary(|| {
        let block = unsafe { read_complex_matrix(block, "onsite block")? };
        let builder = unsafe { builder_mut(handle)? };
        let orbital = orbital(builder, orbital_index)?;
        builder_inner(builder)?
            .set_onsite_block(orbital, block)
            .map_err(|error| AbiError::invalid(error.to_string()))
    })
}

/// Add a scalar hopping. The Hermitian partner is implicit.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_builder_add_hopping(
    handle: *mut ThoulessModelBuilder,
    target: usize,
    source: usize,
    cell_offset: *const i32,
    offset_length: usize,
    amplitude: ThoulessComplex64,
) -> ThoulessStatus {
    boundary(|| {
        let offset = unsafe { borrowed_slice(cell_offset, offset_length, "cell offset")? };
        let builder = unsafe { builder_mut(handle)? };
        let target = orbital(builder, target)?;
        let source = orbital(builder, source)?;
        builder_inner(builder)?
            .add_hopping(
                target,
                source,
                offset.iter().copied(),
                Complex64::from(amplitude),
            )
            .map_err(|error| AbiError::invalid(error.to_string()))
    })
}

/// Add a matrix-valued hopping. The Hermitian partner is implicit.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_builder_add_hopping_block(
    handle: *mut ThoulessModelBuilder,
    target: usize,
    source: usize,
    cell_offset: *const i32,
    offset_length: usize,
    amplitude: ThoulessC64MatrixView,
) -> ThoulessStatus {
    boundary(|| {
        let offset = unsafe { borrowed_slice(cell_offset, offset_length, "cell offset")? };
        let amplitude = unsafe { read_complex_matrix(amplitude, "hopping block")? };
        let builder = unsafe { builder_mut(handle)? };
        let target = orbital(builder, target)?;
        let source = orbital(builder, source)?;
        builder_inner(builder)?
            .add_hopping_block(target, source, offset.iter().copied(), amplitude)
            .map_err(|error| AbiError::invalid(error.to_string()))
    })
}

/// Consume a builder and create an immutable model handle.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_builder_build(
    handle: *mut ThoulessModelBuilder,
    output: *mut *mut ThoulessModel,
) -> ThoulessStatus {
    boundary(|| {
        let builder = unsafe { builder_mut(handle)? };
        let inner = builder
            .inner
            .take()
            .ok_or_else(|| AbiError::invalid("the model builder has already been consumed"))?
            .build()
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe { output_handle(output, ThoulessModel { inner }) }
    })
}

/// Destroy an immutable model. Passing null is a no-op.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_destroy(handle: *mut ThoulessModel) -> ThoulessStatus {
    boundary(|| {
        if !handle.is_null() {
            // SAFETY: the caller transfers the one owned handle back exactly once.
            drop(unsafe { Box::from_raw(handle) });
        }
        Ok(())
    })
}

/// Return the Hamiltonian dimension.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_state_count(
    handle: *const ThoulessModel,
    output: *mut usize,
) -> ThoulessStatus {
    boundary(|| {
        if output.is_null() {
            return Err(AbiError::new(
                ThoulessStatus::NullPointer,
                "state-count output is null",
            ));
        }
        let model = unsafe { model_ref(handle)? };
        // SAFETY: output was checked and points to one writable usize.
        unsafe { *output = model.inner.state_count() };
        Ok(())
    })
}

/// Assemble a Bloch or finite Hamiltonian into caller-owned storage.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_hamiltonian(
    handle: *const ThoulessModel,
    momentum: *const f64,
    momentum_length: usize,
    output: ThoulessC64MatrixMut,
) -> ThoulessStatus {
    boundary(|| {
        let model = unsafe { model_ref(handle)? };
        let momentum = unsafe { borrowed_slice(momentum, momentum_length, "momentum")? };
        let matrix = model
            .inner
            .hamiltonian(momentum)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe { write_complex_matrix(&matrix, output, "Hamiltonian output") }
    })
}

/// Compute ascending eigenvalues and column eigenvectors.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_eigensystem(
    handle: *const ThoulessModel,
    momentum: *const f64,
    momentum_length: usize,
    eigenvalues: *mut f64,
    eigenvalue_capacity: usize,
    eigenvectors: ThoulessC64MatrixMut,
) -> ThoulessStatus {
    boundary(|| {
        let model = unsafe { model_ref(handle)? };
        let momentum = unsafe { borrowed_slice(momentum, momentum_length, "momentum")? };
        let solution = model
            .inner
            .eigensystem(momentum)
            .map_err(|error| AbiError::new(ThoulessStatus::NumericalFailure, error.to_string()))?;
        if eigenvalue_capacity < solution.eigenvalues().len() {
            return Err(AbiError::new(
                ThoulessStatus::BufferTooSmall,
                format!(
                    "eigenvalue output has {eigenvalue_capacity} elements; required {}",
                    solution.eigenvalues().len()
                ),
            ));
        }
        let output =
            unsafe { borrowed_mut_slice(eigenvalues, eigenvalue_capacity, "eigenvalues")? };
        output[..solution.eigenvalues().len()].copy_from_slice(solution.eigenvalues());
        unsafe { write_complex_matrix(solution.eigenvectors(), eigenvectors, "eigenvector output") }
    })
}

/// Extract complete cells into an open finite model.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_finite_cluster(
    handle: *const ThoulessModel,
    cells: *const i32,
    cell_count: usize,
    cell_dimension: usize,
    output: *mut *mut ThoulessModel,
) -> ThoulessStatus {
    boundary(|| {
        let model = unsafe { model_ref(handle)? };
        let length = cell_count.checked_mul(cell_dimension).ok_or_else(|| {
            AbiError::new(ThoulessStatus::ResourceExhausted, "cell shape overflow")
        })?;
        let raw_cells = unsafe { borrowed_slice(cells, length, "finite cells")? };
        let cells = if cell_dimension == 0 {
            vec![Vec::new(); cell_count]
        } else {
            raw_cells
                .chunks(cell_dimension)
                .map(<[i32]>::to_vec)
                .collect::<Vec<_>>()
        };
        let geometry = make_finite_cluster(&model.inner, &cells)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe {
            output_handle(
                output,
                ThoulessModel {
                    inner: geometry.into_model(),
                },
            )
        }
    })
}

/// Extract arbitrary `(cell, orbital)` sites into an open finite model.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_finite_geometry(
    handle: *const ThoulessModel,
    cells: *const i32,
    orbital_indices: *const usize,
    site_count: usize,
    cell_dimension: usize,
    output: *mut *mut ThoulessModel,
) -> ThoulessStatus {
    boundary(|| {
        let model = unsafe { model_ref(handle)? };
        let length = site_count.checked_mul(cell_dimension).ok_or_else(|| {
            AbiError::new(ThoulessStatus::ResourceExhausted, "site shape overflow")
        })?;
        let cells = unsafe { borrowed_slice(cells, length, "finite site cells")? };
        let orbitals =
            unsafe { borrowed_slice(orbital_indices, site_count, "finite site orbitals")? };
        let sites = (0..site_count)
            .map(|site| {
                FiniteSite::new(
                    cells[site * cell_dimension..(site + 1) * cell_dimension]
                        .iter()
                        .copied(),
                    orbitals[site],
                )
            })
            .collect::<Vec<_>>();
        let geometry = make_finite_geometry(&model.inner, &sites)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe {
            output_handle(
                output,
                ThoulessModel {
                    inner: geometry.into_model(),
                },
            )
        }
    })
}

/// Remove selected orbitals and return a new immutable model.
#[no_mangle]
pub unsafe extern "C" fn thouless_model_remove_orbitals(
    handle: *const ThoulessModel,
    removed: *const usize,
    removed_count: usize,
    output: *mut *mut ThoulessModel,
) -> ThoulessStatus {
    boundary(|| {
        let model = unsafe { model_ref(handle)? };
        let removed = unsafe { borrowed_slice(removed, removed_count, "removed orbitals")? };
        let transformed = remove_orbitals(&model.inner, removed)
            .map_err(|error| AbiError::invalid(error.to_string()))?;
        unsafe { output_handle(output, ThoulessModel { inner: transformed }) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_handle_round_trip_and_destruction() {
        // SAFETY: every pointer below comes from a live Rust allocation with
        // the exact extent described to the ABI.
        unsafe {
            let primitive = [1.0];
            let axes = [0usize];
            let mut builder = std::ptr::null_mut();
            assert_eq!(
                thouless_model_builder_create(
                    ThoulessF64MatrixView {
                        data: primitive.as_ptr(),
                        rows: 1,
                        columns: 1,
                        row_stride: 1,
                        column_stride: 1,
                    },
                    axes.as_ptr(),
                    axes.len(),
                    &mut builder,
                ),
                ThoulessStatus::Success
            );
            let mut orbital = usize::MAX;
            let label = b"s";
            let position = [0.0];
            assert_eq!(
                thouless_model_builder_add_orbital(
                    builder,
                    label.as_ptr(),
                    label.len(),
                    position.as_ptr(),
                    position.len(),
                    1,
                    &mut orbital,
                ),
                ThoulessStatus::Success
            );
            assert_eq!(
                thouless_model_builder_add_hopping(
                    builder,
                    orbital,
                    orbital,
                    [1].as_ptr(),
                    1,
                    ThoulessComplex64 { re: -1.0, im: 0.0 },
                ),
                ThoulessStatus::Success
            );
            let mut model = std::ptr::null_mut();
            assert_eq!(
                thouless_model_builder_build(builder, &mut model),
                ThoulessStatus::Success
            );
            let mut values = [0.0];
            let mut vectors = [ThoulessComplex64::default()];
            assert_eq!(
                thouless_model_eigensystem(
                    model,
                    [0.0].as_ptr(),
                    1,
                    values.as_mut_ptr(),
                    values.len(),
                    ThoulessC64MatrixMut {
                        data: vectors.as_mut_ptr(),
                        rows: 1,
                        columns: 1,
                        row_stride: 1,
                        column_stride: 1,
                    },
                ),
                ThoulessStatus::Success
            );
            assert!((values[0] + 2.0).abs() <= 1.0e-12);
            let address = model as usize;
            let workers = (0..8)
                .map(|_| {
                    std::thread::spawn(move || {
                        let model = address as *const ThoulessModel;
                        for _ in 0..50 {
                            let mut count = 0;
                            assert_eq!(
                                thouless_model_state_count(model, &mut count),
                                ThoulessStatus::Success
                            );
                            assert_eq!(count, 1);
                            let mut hamiltonian = [ThoulessComplex64::default()];
                            assert_eq!(
                                thouless_model_hamiltonian(
                                    model,
                                    [0.0].as_ptr(),
                                    1,
                                    ThoulessC64MatrixMut {
                                        data: hamiltonian.as_mut_ptr(),
                                        rows: 1,
                                        columns: 1,
                                        row_stride: 1,
                                        column_stride: 1,
                                    },
                                ),
                                ThoulessStatus::Success
                            );
                            assert!((hamiltonian[0].re + 2.0).abs() <= 1.0e-12);
                        }
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().expect("concurrent immutable model call");
            }
            assert_eq!(thouless_model_destroy(model), ThoulessStatus::Success);
            assert_eq!(
                thouless_model_builder_destroy(builder),
                ThoulessStatus::Success
            );
        }
    }
}
