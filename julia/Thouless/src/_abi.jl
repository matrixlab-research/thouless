const ABI_VERSION_MAJOR = UInt32(1)
const ABI_VERSION_MINOR = UInt32(1)
const STATUS_SUCCESS = Cint(0)
const STATUS_BUFFER_TOO_SMALL = Cint(8)

struct ThoulessError <: Exception
    status::Int32
    message::String
end

Base.showerror(io::IO, error::ThoulessError) =
    print(io, "Thouless ABI error ", error.status, ": ", error.message)

struct _C64
    re::Cdouble
    im::Cdouble
end

_C64(value::Number) = _C64(real(value), imag(value))
Base.ComplexF64(value::_C64) = ComplexF64(value.re, value.im)

struct _F64View
    data::Ptr{Cdouble}
    rows::Csize_t
    columns::Csize_t
    row_stride::Csize_t
    column_stride::Csize_t
end

struct _C64View
    data::Ptr{ComplexF64}
    rows::Csize_t
    columns::Csize_t
    row_stride::Csize_t
    column_stride::Csize_t
end

struct _F64Mut
    data::Ptr{Cdouble}
    rows::Csize_t
    columns::Csize_t
    row_stride::Csize_t
    column_stride::Csize_t
end

struct _C64Mut
    data::Ptr{ComplexF64}
    rows::Csize_t
    columns::Csize_t
    row_stride::Csize_t
    column_stride::Csize_t
end

struct _C64Tensor3View
    data::Ptr{ComplexF64}
    matrices::Csize_t
    rows::Csize_t
    columns::Csize_t
    matrix_stride::Csize_t
    row_stride::Csize_t
    column_stride::Csize_t
end

struct _C64Tensor3Mut
    data::Ptr{ComplexF64}
    matrices::Csize_t
    rows::Csize_t
    columns::Csize_t
    matrix_stride::Csize_t
    row_stride::Csize_t
    column_stride::Csize_t
end

struct _LeadView
    cell_hamiltonian::_C64View
    inter_cell_hopping::_C64View
    coupling::_C64View
end

function _platform_library_name()
    Sys.iswindows() && return "thouless_capi.dll"
    Sys.isapple() && return "libthouless_capi.dylib"
    return "libthouless_capi.so"
end

function _library()
    configured = get(ENV, "THOULESS_LIBRARY", "")
    !isempty(configured) && return abspath(configured)
    bundled = normpath(joinpath(@__DIR__, "..", "deps", "usr", "lib", _platform_library_name()))
    isfile(bundled) && return bundled
    throw(
        ArgumentError(
            "Thouless C library not found; set THOULESS_LIBRARY or install the " *
            "release library at $bundled",
        ),
    )
end

function abi_version()
    version = ccall((:thouless_abi_version, _library()), UInt32, ())
    major = version >> 16
    minor = version & UInt32(0xffff)
    major == ABI_VERSION_MAJOR ||
        throw(ArgumentError("unsupported Thouless ABI major version $major"))
    return (major=major, minor=minor)
end

function _last_error()
    byte_count = ccall((:thouless_last_error_length, _library()), Csize_t, ())
    byte_count == 0 && return ""
    bytes = Vector{UInt8}(undef, byte_count + 1)
    status = GC.@preserve bytes ccall(
        (:thouless_last_error_copy, _library()),
        Cint,
        (Ptr{Cchar}, Csize_t),
        pointer(bytes),
        length(bytes),
    )
    status == STATUS_SUCCESS || return "failed to retrieve the native error"
    return String(bytes[1:byte_count])
end

function _check(status::Integer)
    status == STATUS_SUCCESS && return nothing
    throw(ThoulessError(Int32(status), _last_error()))
end

_real_matrix(value) = Matrix{Float64}(value)
_complex_matrix(value) = Matrix{ComplexF64}(value)

_f64_view(matrix::Matrix{Float64}) =
    _F64View(pointer(matrix), size(matrix, 1), size(matrix, 2), 1, stride(matrix, 2))
_c64_view(matrix::Matrix{ComplexF64}) =
    _C64View(pointer(matrix), size(matrix, 1), size(matrix, 2), 1, stride(matrix, 2))
_f64_mut(matrix::Matrix{Float64}) =
    _F64Mut(pointer(matrix), size(matrix, 1), size(matrix, 2), 1, stride(matrix, 2))
_c64_mut(matrix::Matrix{ComplexF64}) =
    _C64Mut(pointer(matrix), size(matrix, 1), size(matrix, 2), 1, stride(matrix, 2))

function _c64_tensor_view(tensor::Array{ComplexF64,3})
    rows, columns, matrices = size(tensor)
    _C64Tensor3View(pointer(tensor), matrices, rows, columns, rows * columns, 1, rows)
end

function _c64_tensor_mut(tensor::Array{ComplexF64,3})
    rows, columns, matrices = size(tensor)
    _C64Tensor3Mut(pointer(tensor), matrices, rows, columns, rows * columns, 1, rows)
end

_row_major(matrix::AbstractMatrix{T}) where {T} = vec(permutedims(Matrix{T}(matrix)))
