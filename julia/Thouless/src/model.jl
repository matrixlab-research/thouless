struct Lattice
    primitive_vectors::Matrix{Float64}
    periodic_axes::Vector{Int}

    function Lattice(primitive_vectors, periodic_axes)
        primitive = _real_matrix(primitive_vectors)
        axes = Int.(periodic_axes)
        isempty(axes) || all(axis -> 1 <= axis <= size(primitive, 1), axes) ||
            throw(ArgumentError("periodic axes must use Julia's one-based row indices"))
        new(primitive, axes)
    end
end

mutable struct ModelBuilder
    pointer::Ptr{Cvoid}
    consumed::Bool
end

mutable struct Model
    pointer::Ptr{Cvoid}
end

function _destroy_builder!(builder::ModelBuilder)
    if builder.pointer != C_NULL
        ccall(
            (:thouless_model_builder_destroy, _library()),
            Cint,
            (Ptr{Cvoid},),
            builder.pointer,
        )
        builder.pointer = C_NULL
    end
    nothing
end

function _destroy_model!(model::Model)
    if model.pointer != C_NULL
        ccall(
            (:thouless_model_destroy, _library()),
            Cint,
            (Ptr{Cvoid},),
            model.pointer,
        )
        model.pointer = C_NULL
    end
    nothing
end

function ModelBuilder(lattice::Lattice)
    abi_version()
    primitive = lattice.primitive_vectors
    axes = Csize_t.(lattice.periodic_axes .- 1)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve primitive axes output ccall(
        (:thouless_model_builder_create, _library()),
        Cint,
        (_F64View, Ptr{Csize_t}, Csize_t, Ref{Ptr{Cvoid}}),
        _f64_view(primitive),
        pointer(axes),
        length(axes),
        output,
    )
    _check(status)
    builder = ModelBuilder(output[], false)
    finalizer(_destroy_builder!, builder)
    return builder
end

function _live(builder::ModelBuilder)
    builder.pointer != C_NULL || throw(ArgumentError("model builder is closed"))
    builder.consumed && throw(ArgumentError("model builder has already been consumed"))
    return builder.pointer
end

function _live(model::Model)
    model.pointer != C_NULL || throw(ArgumentError("model is closed"))
    return model.pointer
end

function add_orbital!(
    builder::ModelBuilder,
    label::AbstractString,
    reduced_position;
    degrees_of_freedom::Integer=1,
)
    bytes = Vector{UInt8}(codeunits(String(label)))
    position = Float64.(collect(reduced_position))
    output = Ref{Csize_t}()
    status = GC.@preserve bytes position output ccall(
        (:thouless_model_builder_add_orbital, _library()),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Csize_t, Ptr{Cdouble}, Csize_t, Csize_t, Ref{Csize_t}),
        _live(builder),
        pointer(bytes),
        length(bytes),
        pointer(position),
        length(position),
        degrees_of_freedom,
        output,
    )
    _check(status)
    return Int(output[]) + 1
end

function set_onsite!(builder::ModelBuilder, orbital::Integer, energy::Real)
    status = ccall(
        (:thouless_model_builder_set_onsite, _library()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Cdouble),
        _live(builder),
        orbital - 1,
        energy,
    )
    _check(status)
    return builder
end

function set_onsite!(builder::ModelBuilder, orbital::Integer, block::AbstractMatrix)
    matrix = _complex_matrix(block)
    status = GC.@preserve matrix ccall(
        (:thouless_model_builder_set_onsite_block, _library()),
        Cint,
        (Ptr{Cvoid}, Csize_t, _C64View),
        _live(builder),
        orbital - 1,
        _c64_view(matrix),
    )
    _check(status)
    return builder
end

function add_hopping!(
    builder::ModelBuilder,
    target::Integer,
    source::Integer,
    cell_offset,
    amplitude::Number,
)
    offset = Cint.(collect(cell_offset))
    status = GC.@preserve offset ccall(
        (:thouless_model_builder_add_hopping, _library()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cint}, Csize_t, _C64),
        _live(builder),
        target - 1,
        source - 1,
        pointer(offset),
        length(offset),
        _C64(amplitude),
    )
    _check(status)
    return builder
end

function add_hopping!(
    builder::ModelBuilder,
    target::Integer,
    source::Integer,
    cell_offset,
    amplitude::AbstractMatrix,
)
    offset = Cint.(collect(cell_offset))
    matrix = _complex_matrix(amplitude)
    status = GC.@preserve offset matrix ccall(
        (:thouless_model_builder_add_hopping_block, _library()),
        Cint,
        (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cint}, Csize_t, _C64View),
        _live(builder),
        target - 1,
        source - 1,
        pointer(offset),
        length(offset),
        _c64_view(matrix),
    )
    _check(status)
    return builder
end

function build(builder::ModelBuilder)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(
        (:thouless_model_builder_build, _library()),
        Cint,
        (Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        _live(builder),
        output,
    )
    _check(status)
    builder.consumed = true
    model = Model(output[])
    finalizer(_destroy_model!, model)
    return model
end

function state_count(model::Model)
    output = Ref{Csize_t}()
    status = ccall(
        (:thouless_model_state_count, _library()),
        Cint,
        (Ptr{Cvoid}, Ref{Csize_t}),
        _live(model),
        output,
    )
    _check(status)
    return Int(output[])
end

function hamiltonian(model::Model, momentum=Float64[])
    point = Float64.(collect(momentum))
    count = state_count(model)
    output = Matrix{ComplexF64}(undef, count, count)
    status = GC.@preserve point output ccall(
        (:thouless_model_hamiltonian, _library()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, _C64Mut),
        _live(model),
        pointer(point),
        length(point),
        _c64_mut(output),
    )
    _check(status)
    return output
end

function eigensystem(model::Model, momentum=Float64[])
    point = Float64.(collect(momentum))
    count = state_count(model)
    values = Vector{Float64}(undef, count)
    vectors = Matrix{ComplexF64}(undef, count, count)
    status = GC.@preserve point values vectors ccall(
        (:thouless_model_eigensystem, _library()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t, _C64Mut),
        _live(model),
        pointer(point),
        length(point),
        pointer(values),
        length(values),
        _c64_mut(vectors),
    )
    _check(status)
    return (values=values, vectors=vectors)
end

function _model_from_pointer(pointer::Ptr{Cvoid})
    model = Model(pointer)
    finalizer(_destroy_model!, model)
    return model
end
