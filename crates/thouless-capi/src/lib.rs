//! Stable language-neutral ABI for the Thouless scientific core.
//!
//! Every exported entry point contains Rust panics, returns a stable status
//! code, and records a thread-local UTF-8 diagnostic. Inputs are borrowed for
//! the duration of a call. Outputs are copied into caller-owned storage.
//!
//! # Safety
//!
//! Except for scalar version and error-length queries, exported functions are
//! unsafe from Rust: every non-null pointer must address the documented
//! readable or writable extent for the duration of the call. Aliasing caller
//! output with concurrently accessed storage is forbidden.

#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use thouless::{Complex64, ComplexMatrix, RealMatrix};

mod model;
mod workflows;

pub use model::*;
pub use workflows::*;

/// ABI contract encoded as `major << 16 | minor`.
pub const THOULESS_ABI_VERSION: u32 = 1 << 16;

/// Stable status returned by every fallible C ABI function.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThoulessStatus {
    /// The operation completed successfully.
    Success = 0,
    /// A required pointer was null.
    NullPointer = 1,
    /// A scientific scalar or index was invalid.
    InvalidArgument = 2,
    /// An array shape, stride, or output capacity was incompatible.
    ShapeMismatch = 3,
    /// A numerical solve or factorization failed.
    NumericalFailure = 4,
    /// The requested backend or feature is unsupported.
    Unsupported = 5,
    /// The requested allocation or dimension exceeds a checked boundary.
    ResourceExhausted = 6,
    /// A Rust panic was contained at the ABI boundary.
    Panic = 7,
    /// Caller-owned output storage was too small.
    BufferTooSmall = 8,
    /// A Rust invariant failed without panicking.
    InternalError = 9,
}

/// C-compatible complex binary64 scalar.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThoulessComplex64 {
    pub re: f64,
    pub im: f64,
}

impl From<ThoulessComplex64> for Complex64 {
    fn from(value: ThoulessComplex64) -> Self {
        Self::new(value.re, value.im)
    }
}

impl From<Complex64> for ThoulessComplex64 {
    fn from(value: Complex64) -> Self {
        Self {
            re: value.re,
            im: value.im,
        }
    }
}

/// Borrowed real matrix with element strides.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ThoulessF64MatrixView {
    pub data: *const f64,
    pub rows: usize,
    pub columns: usize,
    pub row_stride: usize,
    pub column_stride: usize,
}

/// Borrowed complex matrix with element strides.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ThoulessC64MatrixView {
    pub data: *const ThoulessComplex64,
    pub rows: usize,
    pub columns: usize,
    pub row_stride: usize,
    pub column_stride: usize,
}

/// Caller-owned mutable real matrix with element strides.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ThoulessF64MatrixMut {
    pub data: *mut f64,
    pub rows: usize,
    pub columns: usize,
    pub row_stride: usize,
    pub column_stride: usize,
}

/// Caller-owned mutable complex matrix with element strides.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ThoulessC64MatrixMut {
    pub data: *mut ThoulessComplex64,
    pub rows: usize,
    pub columns: usize,
    pub row_stride: usize,
    pub column_stride: usize,
}

/// Borrowed stack of complex matrices with element strides.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ThoulessC64Tensor3View {
    pub data: *const ThoulessComplex64,
    pub matrices: usize,
    pub rows: usize,
    pub columns: usize,
    pub matrix_stride: usize,
    pub row_stride: usize,
    pub column_stride: usize,
}

/// Caller-owned mutable stack of complex matrices with element strides.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ThoulessC64Tensor3Mut {
    pub data: *mut ThoulessComplex64,
    pub matrices: usize,
    pub rows: usize,
    pub columns: usize,
    pub matrix_stride: usize,
    pub row_stride: usize,
    pub column_stride: usize,
}

/// One borrowed periodic lead and device coupling.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ThoulessLeadView {
    pub cell_hamiltonian: ThoulessC64MatrixView,
    pub inter_cell_hopping: ThoulessC64MatrixView,
    pub coupling: ThoulessC64MatrixView,
}

#[derive(Debug)]
pub(crate) struct AbiError {
    pub status: ThoulessStatus,
    pub message: String,
}

impl AbiError {
    pub(crate) fn new(status: ThoulessStatus, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::new(ThoulessStatus::InvalidArgument, message)
    }

    pub(crate) fn shape(message: impl Into<String>) -> Self {
        Self::new(ThoulessStatus::ShapeMismatch, message)
    }
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

pub(crate) fn set_last_error(message: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
}

pub(crate) fn boundary(operation: impl FnOnce() -> Result<(), AbiError>) -> ThoulessStatus {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => {
            set_last_error("");
            ThoulessStatus::Success
        }
        Ok(Err(error)) => {
            set_last_error(error.message);
            error.status
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|value| (*value).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "Rust panic without a string payload".to_owned());
            set_last_error(message);
            ThoulessStatus::Panic
        }
    }
}

pub(crate) unsafe fn borrowed_slice<'a, T>(
    data: *const T,
    length: usize,
    name: &str,
) -> Result<&'a [T], AbiError> {
    if length == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(AbiError::new(
            ThoulessStatus::NullPointer,
            format!("{name} is null"),
        ));
    }
    // SAFETY: the caller promises a readable allocation containing `length`
    // elements for the duration of this call.
    Ok(unsafe { slice::from_raw_parts(data, length) })
}

pub(crate) unsafe fn borrowed_mut_slice<'a, T>(
    data: *mut T,
    length: usize,
    name: &str,
) -> Result<&'a mut [T], AbiError> {
    if length == 0 {
        return Ok(&mut []);
    }
    if data.is_null() {
        return Err(AbiError::new(
            ThoulessStatus::NullPointer,
            format!("{name} is null"),
        ));
    }
    // SAFETY: the caller promises a writable allocation containing `length`
    // elements and exclusive access for the duration of this call.
    Ok(unsafe { slice::from_raw_parts_mut(data, length) })
}

fn matrix_extent(
    rows: usize,
    columns: usize,
    row_stride: usize,
    column_stride: usize,
) -> Result<usize, AbiError> {
    if rows == 0 || columns == 0 {
        return Ok(0);
    }
    (rows - 1)
        .checked_mul(row_stride)
        .and_then(|row| {
            (columns - 1)
                .checked_mul(column_stride)
                .and_then(|column| row.checked_add(column))
        })
        .and_then(|last| last.checked_add(1))
        .ok_or_else(|| AbiError::new(ThoulessStatus::ResourceExhausted, "matrix extent overflow"))
}

fn tensor3_extent(
    matrices: usize,
    rows: usize,
    columns: usize,
    matrix_stride: usize,
    row_stride: usize,
    column_stride: usize,
) -> Result<usize, AbiError> {
    if matrices == 0 || rows == 0 || columns == 0 {
        return Ok(0);
    }
    (matrices - 1)
        .checked_mul(matrix_stride)
        .and_then(|matrix| {
            (rows - 1)
                .checked_mul(row_stride)
                .and_then(|row| matrix.checked_add(row))
        })
        .and_then(|matrix_row| {
            (columns - 1)
                .checked_mul(column_stride)
                .and_then(|column| matrix_row.checked_add(column))
        })
        .and_then(|last| last.checked_add(1))
        .ok_or_else(|| AbiError::new(ThoulessStatus::ResourceExhausted, "tensor extent overflow"))
}

pub(crate) unsafe fn read_real_matrix(
    view: ThoulessF64MatrixView,
    name: &str,
) -> Result<RealMatrix, AbiError> {
    let extent = matrix_extent(view.rows, view.columns, view.row_stride, view.column_stride)?;
    let source = unsafe { borrowed_slice(view.data, extent, name)? };
    let values = (0..view.rows)
        .flat_map(|row| {
            (0..view.columns)
                .map(move |column| source[row * view.row_stride + column * view.column_stride])
        })
        .collect();
    RealMatrix::new(view.rows, view.columns, values)
        .map_err(|error| AbiError::shape(error.to_string()))
}

pub(crate) unsafe fn read_complex_matrix(
    view: ThoulessC64MatrixView,
    name: &str,
) -> Result<ComplexMatrix, AbiError> {
    let extent = matrix_extent(view.rows, view.columns, view.row_stride, view.column_stride)?;
    let source = unsafe { borrowed_slice(view.data, extent, name)? };
    let values = (0..view.rows)
        .flat_map(|row| {
            (0..view.columns).map(move |column| {
                Complex64::from(source[row * view.row_stride + column * view.column_stride])
            })
        })
        .collect();
    ComplexMatrix::new(view.rows, view.columns, values)
        .map_err(|error| AbiError::shape(error.to_string()))
}

pub(crate) unsafe fn write_real_matrix(
    matrix: &RealMatrix,
    output: ThoulessF64MatrixMut,
    name: &str,
) -> Result<(), AbiError> {
    if matrix.shape() != (output.rows, output.columns) {
        return Err(AbiError::new(
            ThoulessStatus::BufferTooSmall,
            format!(
                "{name} has shape {}x{}; required {}x{}",
                output.rows,
                output.columns,
                matrix.rows(),
                matrix.columns()
            ),
        ));
    }
    let extent = matrix_extent(
        output.rows,
        output.columns,
        output.row_stride,
        output.column_stride,
    )?;
    let destination = unsafe { borrowed_mut_slice(output.data, extent, name)? };
    for row in 0..output.rows {
        for column in 0..output.columns {
            destination[row * output.row_stride + column * output.column_stride] =
                matrix.as_slice()[row * output.columns + column];
        }
    }
    Ok(())
}

pub(crate) unsafe fn write_complex_matrix(
    matrix: &ComplexMatrix,
    output: ThoulessC64MatrixMut,
    name: &str,
) -> Result<(), AbiError> {
    if matrix.shape() != (output.rows, output.columns) {
        return Err(AbiError::new(
            ThoulessStatus::BufferTooSmall,
            format!(
                "{name} has shape {}x{}; required {}x{}",
                output.rows,
                output.columns,
                matrix.rows(),
                matrix.columns()
            ),
        ));
    }
    let extent = matrix_extent(
        output.rows,
        output.columns,
        output.row_stride,
        output.column_stride,
    )?;
    let destination = unsafe { borrowed_mut_slice(output.data, extent, name)? };
    for row in 0..output.rows {
        for column in 0..output.columns {
            destination[row * output.row_stride + column * output.column_stride] =
                matrix.as_slice()[row * output.columns + column].into();
        }
    }
    Ok(())
}

pub(crate) unsafe fn read_complex_tensor3(
    view: ThoulessC64Tensor3View,
    name: &str,
) -> Result<Vec<ComplexMatrix>, AbiError> {
    let extent = tensor3_extent(
        view.matrices,
        view.rows,
        view.columns,
        view.matrix_stride,
        view.row_stride,
        view.column_stride,
    )?;
    let source = unsafe { borrowed_slice(view.data, extent, name)? };
    (0..view.matrices)
        .map(|matrix| {
            let values = (0..view.rows)
                .flat_map(|row| {
                    (0..view.columns).map(move |column| {
                        Complex64::from(
                            source[matrix * view.matrix_stride
                                + row * view.row_stride
                                + column * view.column_stride],
                        )
                    })
                })
                .collect();
            ComplexMatrix::new(view.rows, view.columns, values)
                .map_err(|error| AbiError::shape(error.to_string()))
        })
        .collect()
}

pub(crate) unsafe fn write_complex_tensor3(
    matrices: &[ComplexMatrix],
    output: ThoulessC64Tensor3Mut,
    name: &str,
) -> Result<(), AbiError> {
    if matrices.len() != output.matrices
        || matrices
            .iter()
            .any(|matrix| matrix.shape() != (output.rows, output.columns))
    {
        return Err(AbiError::new(
            ThoulessStatus::BufferTooSmall,
            format!(
                "{name} has shape {}x{}x{}; required a common exact result shape",
                output.matrices, output.rows, output.columns
            ),
        ));
    }
    let extent = tensor3_extent(
        output.matrices,
        output.rows,
        output.columns,
        output.matrix_stride,
        output.row_stride,
        output.column_stride,
    )?;
    let destination = unsafe { borrowed_mut_slice(output.data, extent, name)? };
    for (matrix_index, matrix) in matrices.iter().enumerate() {
        for row in 0..output.rows {
            for column in 0..output.columns {
                destination[matrix_index * output.matrix_stride
                    + row * output.row_stride
                    + column * output.column_stride] =
                    matrix.as_slice()[row * output.columns + column].into();
            }
        }
    }
    Ok(())
}

/// Return the encoded ABI version.
#[no_mangle]
pub extern "C" fn thouless_abi_version() -> u32 {
    THOULESS_ABI_VERSION
}

/// Return the current thread's last-error byte length, excluding NUL.
#[no_mangle]
pub extern "C" fn thouless_last_error_length() -> usize {
    LAST_ERROR.with(|slot| slot.borrow().len())
}

/// Copy the current thread's UTF-8 last error and a trailing NUL.
#[no_mangle]
pub unsafe extern "C" fn thouless_last_error_copy(
    buffer: *mut c_char,
    capacity: usize,
) -> ThoulessStatus {
    boundary(|| {
        let message = LAST_ERROR.with(|slot| slot.borrow().clone());
        let required = message.len().saturating_add(1);
        if capacity < required {
            return Err(AbiError::new(
                ThoulessStatus::BufferTooSmall,
                format!("error buffer has {capacity} bytes; required {required}"),
            ));
        }
        let destination =
            unsafe { borrowed_mut_slice(buffer.cast::<u8>(), capacity, "error buffer")? };
        destination[..message.len()].copy_from_slice(message.as_bytes());
        destination[message.len()] = 0;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_is_contained_and_reported() {
        let status = boundary(|| -> Result<(), AbiError> { panic!("contained panic") });
        assert_eq!(status, ThoulessStatus::Panic);
        assert_eq!(thouless_last_error_length(), "contained panic".len());
    }

    #[test]
    fn strided_complex_matrix_round_trips() {
        let input = [
            ThoulessComplex64 { re: 1.0, im: 0.0 },
            ThoulessComplex64 { re: 99.0, im: 0.0 },
            ThoulessComplex64 { re: 2.0, im: -1.0 },
            ThoulessComplex64 { re: 3.0, im: 1.0 },
            ThoulessComplex64 { re: 99.0, im: 0.0 },
            ThoulessComplex64 { re: 4.0, im: 0.0 },
        ];
        let matrix = unsafe {
            read_complex_matrix(
                ThoulessC64MatrixView {
                    data: input.as_ptr(),
                    rows: 2,
                    columns: 2,
                    row_stride: 3,
                    column_stride: 2,
                },
                "input",
            )
        }
        .unwrap();
        assert_eq!(matrix.as_slice()[1], Complex64::new(2.0, -1.0));
        assert_eq!(matrix.as_slice()[2], Complex64::new(3.0, 1.0));
    }
}
