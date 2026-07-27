function _ad_affine_projector_value_and_grad(
    base,
    directions,
    parameters;
    occupied,
    target,
    minimum_gap=1.0e-8,
)
    base_matrix = _complex_matrix(base)
    direction_matrices = [_complex_matrix(direction) for direction in directions]
    isempty(direction_matrices) &&
        throw(ArgumentError("at least one parameter direction is required"))
    all(size(direction) == size(base_matrix) for direction in direction_matrices) ||
        throw(ArgumentError("all parameter directions must match the base matrix"))
    direction_tensor = cat(direction_matrices...; dims=3)
    parameter_values = Float64.(collect(parameters))
    target_matrix = _complex_matrix(target)
    gradient = Vector{Float64}(undef, length(parameter_values))
    value = Ref{Cdouble}()
    status = GC.@preserve base_matrix direction_tensor parameter_values target_matrix gradient value ccall(
        (:thouless_ad_affine_projector_value_and_grad, _library()),
        Cint,
        (
            _C64View,
            _C64Tensor3View,
            Ptr{Cdouble},
            Csize_t,
            Csize_t,
            _C64View,
            Cdouble,
            Ref{Cdouble},
            Ptr{Cdouble},
            Csize_t,
        ),
        _c64_view(base_matrix),
        _c64_tensor_view(direction_tensor),
        pointer(parameter_values),
        length(parameter_values),
        Int(occupied),
        _c64_view(target_matrix),
        minimum_gap,
        value,
        pointer(gradient),
        length(gradient),
    )
    _check(status)
    return (value=value[], gradient=gradient)
end

function _hermitian_eigensystem(value)
    matrix = _complex_matrix(value)
    size(matrix, 1) == size(matrix, 2) || throw(ArgumentError("matrix must be square"))
    count = size(matrix, 1)
    values = Vector{Float64}(undef, count)
    vectors = Matrix{ComplexF64}(undef, count, count)
    status = GC.@preserve matrix values vectors ccall(
        (:thouless_hermitian_eigensystem, _library()),
        Cint,
        (_C64View, Ptr{Cdouble}, Csize_t, _C64Mut),
        _c64_view(matrix),
        pointer(values),
        count,
        _c64_mut(vectors),
    )
    _check(status)
    return (values=values, vectors=vectors)
end

function _kpm_rescale(value; strict_margin=0.05, bounds=nothing)
    matrix = _complex_matrix(value)
    output = similar(matrix)
    half_width = Ref{Cdouble}()
    center = Ref{Cdouble}()
    explicit = bounds !== nothing
    lower, upper = explicit ? Float64.(bounds) : (0.0, 0.0)
    status = GC.@preserve matrix output half_width center ccall(
        (:thouless_kpm_rescale_dense, _library()),
        Cint,
        (_C64View, Cdouble, Cuchar, Cdouble, Cdouble, _C64Mut, Ref{Cdouble}, Ref{Cdouble}),
        _c64_view(matrix),
        strict_margin,
        explicit,
        lower,
        upper,
        _c64_mut(output),
        half_width,
        center,
    )
    _check(status)
    return (matrix=output, half_width=half_width[], center=center[])
end

function _lead_bands(cell, hopping, momentum; derivative_order=0)
    cell_matrix = _complex_matrix(cell)
    hopping_matrix = _complex_matrix(hopping)
    count = size(cell_matrix, 1)
    energies = Vector{Float64}(undef, count)
    first = derivative_order >= 1 ? Vector{Float64}(undef, count) : Float64[]
    second = derivative_order >= 2 ? Vector{Float64}(undef, count) : Float64[]
    status = GC.@preserve cell_matrix hopping_matrix energies first second ccall(
        (:thouless_lead_bands, _library()),
        Cint,
        (
            _C64View,
            _C64View,
            Cdouble,
            Csize_t,
            Ptr{Cdouble},
            Csize_t,
            Ptr{Cdouble},
            Csize_t,
            Ptr{Cdouble},
            Csize_t,
        ),
        _c64_view(cell_matrix),
        _c64_view(hopping_matrix),
        momentum,
        derivative_order,
        pointer(energies),
        length(energies),
        pointer(first),
        length(first),
        pointer(second),
        length(second),
    )
    _check(status)
    return (
        energies=energies,
        first_derivatives=derivative_order >= 1 ? first : nothing,
        second_derivatives=derivative_order >= 2 ? second : nothing,
    )
end

function _finite_cluster(model::Model, cells)
    matrix = Matrix{Cint}(cells)
    raw = _row_major(matrix)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve raw output ccall(
        (:thouless_model_finite_cluster, _library()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cint}, Csize_t, Csize_t, Ref{Ptr{Cvoid}}),
        _live(model),
        pointer(raw),
        size(matrix, 1),
        size(matrix, 2),
        output,
    )
    _check(status)
    return _model_from_pointer(output[])
end

function _finite_geometry(model::Model, cells, orbitals)
    matrix = Matrix{Cint}(cells)
    raw = _row_major(matrix)
    orbital_indices = Csize_t.(collect(orbitals) .- 1)
    length(orbital_indices) == size(matrix, 1) ||
        throw(ArgumentError("one orbital index is required per site"))
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve raw orbital_indices output ccall(
        (:thouless_model_finite_geometry, _library()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cint}, Ptr{Csize_t}, Csize_t, Csize_t, Ref{Ptr{Cvoid}}),
        _live(model),
        pointer(raw),
        pointer(orbital_indices),
        size(matrix, 1),
        size(matrix, 2),
        output,
    )
    _check(status)
    return _model_from_pointer(output[])
end

function _remove_orbitals(model::Model, removed)
    indices = Csize_t.(collect(removed) .- 1)
    output = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve indices output ccall(
        (:thouless_model_remove_orbitals, _library()),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Ptr{Cvoid}}),
        _live(model),
        pointer(indices),
        length(indices),
        output,
    )
    _check(status)
    return _model_from_pointer(output[])
end

function _bloch_phase(translation, momentum)
    displacement = Int64.(collect(translation))
    point = Float64.(collect(momentum))
    output = Ref{_C64}()
    status = GC.@preserve displacement point output ccall(
        (:thouless_bloch_phase, _library()),
        Cint,
        (Ptr{Int64}, Csize_t, Ptr{Cdouble}, Csize_t, Ref{_C64}),
        pointer(displacement),
        length(displacement),
        pointer(point),
        length(point),
        output,
    )
    _check(status)
    return ComplexF64(output[])
end

function _reciprocal_path(lattice::Lattice, nodes, sample_count::Integer)
    node_matrix = _real_matrix(nodes)
    dimension = length(lattice.periodic_axes)
    size(node_matrix, 2) == dimension ||
        throw(ArgumentError("path nodes must use the periodic dimension"))
    axes = Csize_t.(lattice.periodic_axes .- 1)
    primitive = lattice.primitive_vectors
    points = Matrix{Float64}(undef, sample_count, dimension)
    distances = Vector{Float64}(undef, sample_count)
    node_distances = Vector{Float64}(undef, size(node_matrix, 1))
    status = GC.@preserve primitive node_matrix axes points distances node_distances ccall(
        (:thouless_reciprocal_path, _library()),
        Cint,
        (
            _F64View,
            Ptr{Csize_t},
            Csize_t,
            _F64View,
            Csize_t,
            _F64Mut,
            Ptr{Cdouble},
            Csize_t,
            Ptr{Cdouble},
            Csize_t,
        ),
        _f64_view(primitive),
        pointer(axes),
        length(axes),
        _f64_view(node_matrix),
        sample_count,
        _f64_mut(points),
        pointer(distances),
        length(distances),
        pointer(node_distances),
        length(node_distances),
    )
    _check(status)
    return (points=points, distances=distances, node_distances=node_distances)
end

function _momentum_stencil(dimension::Integer, factors)
    axes = Csize_t[first(factor) - 1 for factor in factors]
    powers = Csize_t[last(factor) for factor in factors]
    term_count = Ref{Csize_t}()
    status = GC.@preserve axes powers term_count ccall(
        (:thouless_continuum_momentum_stencil, _library()),
        Cint,
        (
            Csize_t,
            Ptr{Csize_t},
            Ptr{Csize_t},
            Csize_t,
            Ptr{Cint},
            Ptr{UInt32},
            Ptr{_C64},
            Csize_t,
            Ref{Csize_t},
        ),
        dimension,
        pointer(axes),
        pointer(powers),
        length(axes),
        C_NULL,
        C_NULL,
        C_NULL,
        0,
        term_count,
    )
    status == STATUS_BUFFER_TOO_SMALL || _check(status)
    count = Int(term_count[])
    offsets = Vector{Cint}(undef, count * dimension)
    inverse = Vector{UInt32}(undef, count * dimension)
    weights = Vector{_C64}(undef, count)
    status = GC.@preserve axes powers offsets inverse weights term_count ccall(
        (:thouless_continuum_momentum_stencil, _library()),
        Cint,
        (
            Csize_t,
            Ptr{Csize_t},
            Ptr{Csize_t},
            Csize_t,
            Ptr{Cint},
            Ptr{UInt32},
            Ptr{_C64},
            Csize_t,
            Ref{Csize_t},
        ),
        dimension,
        pointer(axes),
        pointer(powers),
        length(axes),
        pointer(offsets),
        pointer(inverse),
        pointer(weights),
        count,
        term_count,
    )
    _check(status)
    return (
        offsets=permutedims(reshape(Int32.(offsets), dimension, count)),
        inverse_spacing_powers=permutedims(reshape(inverse, dimension, count)),
        weights=ComplexF64.(weights),
    )
end

function _interpolate_density(
    points,
    values,
    reference_starts,
    reference_ends;
    absolute_width,
    samples_per_width=9,
)
    point_matrix = _real_matrix(points)
    density_values = Float64.(collect(values))
    starts = _real_matrix(reference_starts)
    ends = _real_matrix(reference_ends)
    dimension = size(point_matrix, 2)
    shape = Vector{Csize_t}(undef, dimension)
    bounds = Matrix{Float64}(undef, dimension, 2)
    required = Ref{Csize_t}()
    status = GC.@preserve point_matrix density_values starts ends shape bounds required ccall(
        (:thouless_interpolate_density, _library()),
        Cint,
        (
            _F64View,
            Ptr{Cdouble},
            Csize_t,
            _F64View,
            _F64View,
            Cdouble,
            Csize_t,
            Ptr{Csize_t},
            Csize_t,
            _F64Mut,
            Ptr{Cdouble},
            Csize_t,
            Ref{Csize_t},
        ),
        _f64_view(point_matrix),
        pointer(density_values),
        length(density_values),
        _f64_view(starts),
        _f64_view(ends),
        absolute_width,
        samples_per_width,
        pointer(shape),
        length(shape),
        _f64_mut(bounds),
        C_NULL,
        0,
        required,
    )
    status == STATUS_BUFFER_TOO_SMALL || _check(status)
    output = Vector{Float64}(undef, required[])
    status = GC.@preserve point_matrix density_values starts ends shape bounds required output ccall(
        (:thouless_interpolate_density, _library()),
        Cint,
        (
            _F64View,
            Ptr{Cdouble},
            Csize_t,
            _F64View,
            _F64View,
            Cdouble,
            Csize_t,
            Ptr{Csize_t},
            Csize_t,
            _F64Mut,
            Ptr{Cdouble},
            Csize_t,
            Ref{Csize_t},
        ),
        _f64_view(point_matrix),
        pointer(density_values),
        length(density_values),
        _f64_view(starts),
        _f64_view(ends),
        absolute_width,
        samples_per_width,
        pointer(shape),
        length(shape),
        _f64_mut(bounds),
        pointer(output),
        length(output),
        required,
    )
    _check(status)
    return (values=output, shape=Int.(shape), components=1, bounds=bounds)
end

function _stack_frames(frames)
    converted = [_complex_matrix(frame) for frame in frames]
    isempty(converted) && throw(ArgumentError("at least one frame is required"))
    rows, columns = size(first(converted))
    all(size(frame) == (rows, columns) for frame in converted) ||
        throw(ArgumentError("all frames must have the same shape"))
    tensor = Array{ComplexF64}(undef, rows, columns, length(converted))
    for (index, frame) in enumerate(converted)
        tensor[:, :, index] = frame
    end
    return tensor
end

function _wilson_phase(frames)
    tensor = _stack_frames(frames)
    output = Ref{Cdouble}()
    status = GC.@preserve tensor output ccall(
        (:thouless_wilson_phase, _library()),
        Cint,
        (_C64Tensor3View, Ref{Cdouble}),
        _c64_tensor_view(tensor),
        output,
    )
    _check(status)
    return output[]
end

function _chern_numbers(model::Model, samples, plane, occupied)
    sample_counts = Csize_t.(collect(samples))
    directions = Int.(collect(plane))
    length(directions) == 2 || throw(ArgumentError("Chern plane needs two directions"))
    states = Csize_t.(collect(occupied) .- 1)
    spectator_count = prod(
        sample_counts[[
            index for index in eachindex(sample_counts) if
            !(index in (directions[1], directions[2]))
        ]],
    )
    output = Vector{Float64}(undef, spectator_count)
    value_count = Ref{Csize_t}()
    status = GC.@preserve sample_counts states output value_count ccall(
        (:thouless_model_chern_numbers, _library()),
        Cint,
        (
            Ptr{Cvoid},
            Ptr{Csize_t},
            Csize_t,
            Csize_t,
            Csize_t,
            Ptr{Csize_t},
            Csize_t,
            Ptr{Cdouble},
            Csize_t,
            Ref{Csize_t},
        ),
        _live(model),
        pointer(sample_counts),
        length(sample_counts),
        directions[1] - 1,
        directions[2] - 1,
        pointer(states),
        length(states),
        pointer(output),
        length(output),
        value_count,
    )
    _check(status)
    resize!(output, value_count[])
    return output
end

function _quantum_geometric_tensor(hamiltonian, derivatives, occupied)
    matrix = _complex_matrix(hamiltonian)
    tensor = _stack_frames(derivatives)
    states = Csize_t.(collect(occupied) .- 1)
    directions = size(tensor, 3)
    occupied_count = length(states)
    output = Vector{_C64}(undef, directions^2 * occupied_count^2)
    status = GC.@preserve matrix tensor states output ccall(
        (:thouless_quantum_geometric_tensor, _library()),
        Cint,
        (_C64View, _C64Tensor3View, Ptr{Csize_t}, Csize_t, Ptr{_C64}, Csize_t),
        _c64_view(matrix),
        _c64_tensor_view(tensor),
        pointer(states),
        occupied_count,
        pointer(output),
        length(output),
    )
    _check(status)
    result = Array{ComplexF64}(undef, occupied_count, occupied_count, directions, directions)
    raw = ComplexF64.(output)
    cursor = 1
    for first_direction in 1:directions, second_direction in 1:directions
        block = raw[cursor:(cursor + occupied_count^2 - 1)]
        result[:, :, second_direction, first_direction] =
            permutedims(reshape(block, occupied_count, occupied_count))
        cursor += occupied_count^2
    end
    return result
end

function _local_chern_marker(hamiltonian, positions, occupied, cell_area)
    matrix = _complex_matrix(hamiltonian)
    point_matrix = _real_matrix(positions)
    states = Csize_t.(collect(occupied) .- 1)
    output = Vector{Float64}(undef, size(matrix, 1))
    status = GC.@preserve matrix point_matrix states output ccall(
        (:thouless_local_chern_marker, _library()),
        Cint,
        (_C64View, _F64View, Ptr{Csize_t}, Csize_t, Cdouble, Ptr{Cdouble}, Csize_t),
        _c64_view(matrix),
        _f64_view(point_matrix),
        pointer(states),
        length(states),
        cell_area,
        pointer(output),
        length(output),
    )
    _check(status)
    return output
end

function _project_trials(frames, trials; singular_tolerance=1.0e-10)
    tensor = _stack_frames(frames)
    trial_matrix = _complex_matrix(trials)
    output = Array{ComplexF64}(
        undef,
        size(trial_matrix, 1),
        size(tensor, 2),
        size(tensor, 3),
    )
    status = GC.@preserve tensor trial_matrix output ccall(
        (:thouless_wannier_project_trials, _library()),
        Cint,
        (_C64Tensor3View, _C64View, Cdouble, _C64Tensor3Mut),
        _c64_tensor_view(tensor),
        _c64_view(trial_matrix),
        singular_tolerance,
        _c64_tensor_mut(output),
    )
    _check(status)
    return [copy(output[:, :, index]) for index in axes(output, 3)]
end

function _intrinsic_curvature(
    model::Model,
    momentum;
    chemical_potential=0.0,
    temperature=0.0,
    coordinates=:reduced,
    degeneracy_tolerance=1.0e-9,
)
    point = Float64.(collect(momentum))
    output = Matrix{Float64}(undef, length(point), length(point))
    status = GC.@preserve point output ccall(
        (:thouless_model_intrinsic_curvature, _library()),
        Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Cdouble, Cdouble, Cuchar, Cdouble, _F64Mut),
        _live(model),
        pointer(point),
        length(point),
        chemical_potential,
        temperature,
        coordinates === :cartesian,
        degeneracy_tolerance,
        _f64_mut(output),
    )
    _check(status)
    return output
end

function _project_diagonal(states, diagonal)
    matrix = _complex_matrix(states)
    values = Float64.(collect(diagonal))
    output = Matrix{ComplexF64}(undef, size(matrix, 2), size(matrix, 2))
    status = GC.@preserve matrix values output ccall(
        (:thouless_project_diagonal_observable, _library()),
        Cint,
        (_C64View, Ptr{Cdouble}, Csize_t, _C64Mut),
        _c64_view(matrix),
        pointer(values),
        length(values),
        _c64_mut(output),
    )
    _check(status)
    return output
end

struct _NativeLead
    cell::Matrix{ComplexF64}
    hopping::Matrix{ComplexF64}
    coupling::Matrix{ComplexF64}
end

function _transmissions(device, leads, energy; broadening=1.0e-4)
    device_matrix = _complex_matrix(device)
    native = [
        _NativeLead(
            _complex_matrix(lead.cell_hamiltonian),
            _complex_matrix(lead.inter_cell_hopping),
            _complex_matrix(lead.coupling),
        ) for lead in leads
    ]
    views = [
        _LeadView(_c64_view(lead.cell), _c64_view(lead.hopping), _c64_view(lead.coupling)) for
        lead in native
    ]
    output = Matrix{Float64}(undef, length(leads), length(leads))
    status = GC.@preserve device_matrix native views output ccall(
        (:thouless_open_system_transmissions, _library()),
        Cint,
        (_C64View, Ptr{_LeadView}, Csize_t, Cdouble, Cdouble, _F64Mut),
        _c64_view(device_matrix),
        pointer(views),
        length(views),
        energy,
        broadening,
        _f64_mut(output),
    )
    _check(status)
    return output
end

function _lead_modes(cell, hopping)
    cell_matrix = _complex_matrix(cell)
    hopping_matrix = _complex_matrix(hopping)
    count = Ref{Csize_t}()
    incoming = Ref{Csize_t}()
    status = GC.@preserve cell_matrix hopping_matrix count incoming ccall(
        (:thouless_lead_mode_count, _library()),
        Cint,
        (_C64View, _C64View, Ref{Csize_t}, Ref{Csize_t}),
        _c64_view(cell_matrix),
        _c64_view(hopping_matrix),
        count,
        incoming,
    )
    _check(status)
    wave_functions = Matrix{ComplexF64}(undef, size(cell_matrix, 1), count[])
    velocities = Vector{Float64}(undef, count[])
    momenta = Vector{Float64}(undef, count[])
    status = GC.@preserve cell_matrix hopping_matrix wave_functions velocities momenta ccall(
        (:thouless_lead_modes, _library()),
        Cint,
        (_C64View, _C64View, _C64Mut, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t),
        _c64_view(cell_matrix),
        _c64_view(hopping_matrix),
        _c64_mut(wave_functions),
        pointer(velocities),
        length(velocities),
        pointer(momenta),
        length(momenta),
    )
    _check(status)
    return (
        wave_functions=wave_functions,
        velocities=velocities,
        momenta=momenta,
        incoming_count=Int(incoming[]),
    )
end

function _lead_self_energy(cell, hopping, energy=0.0; broadening=1.0e-4, maximum_rank=nothing)
    cell_matrix = _complex_matrix(cell)
    hopping_matrix = _complex_matrix(hopping)
    output = Matrix{ComplexF64}(undef, size(hopping_matrix, 2), size(hopping_matrix, 2))
    status = GC.@preserve cell_matrix hopping_matrix output ccall(
        (:thouless_lead_self_energy, _library()),
        Cint,
        (_C64View, _C64View, Cdouble, Cdouble, Csize_t, Cuchar, _C64Mut),
        _c64_view(cell_matrix),
        _c64_view(hopping_matrix),
        energy,
        broadening,
        maximum_rank === nothing ? 0 : maximum_rank,
        maximum_rank !== nothing,
        _c64_mut(output),
    )
    _check(status)
    return output
end

function _validate_chiral(matrix, chiral)
    value = _complex_matrix(matrix)
    operator = _complex_matrix(chiral)
    count = Ref{Csize_t}()
    status = GC.@preserve value operator count ccall(
        (:thouless_validate_chiral_symmetry, _library()),
        Cint,
        (_C64View, _C64View, Ref{Csize_t}),
        _c64_view(value),
        _c64_view(operator),
        count,
    )
    _check(status)
    return Int(count[])
end

function _particle_hole_basis(wave_functions, particle_hole)
    vectors = _complex_matrix(wave_functions)
    operator = _complex_matrix(particle_hole)
    output = similar(vectors)
    ordering = Vector{Csize_t}(undef, size(vectors, 2))
    status = GC.@preserve vectors operator output ordering ccall(
        (:thouless_particle_hole_basis, _library()),
        Cint,
        (_C64View, _C64View, _C64Mut, Ptr{Csize_t}, Csize_t),
        _c64_view(vectors),
        _c64_view(operator),
        _c64_mut(output),
        pointer(ordering),
        length(ordering),
    )
    _check(status)
    return (wave_functions=output, ordering=Int.(ordering) .+ 1)
end

const _SYMMETRY_CLASSES = Dict(
    :A => 0,
    :AI => 1,
    :AII => 2,
    :AIII => 3,
    :BDI => 4,
    :CII => 5,
    :D => 6,
    :DIII => 7,
    :C => 8,
    :CI => 9,
)

function _gaussian_matrix(dimension, symmetry, variance, real_components, imaginary_components)
    real_values = Float64.(collect(real_components))
    imaginary_values = Float64.(collect(imaginary_components))
    output = Matrix{ComplexF64}(undef, dimension, dimension)
    class = get(_SYMMETRY_CLASSES, Symbol(uppercase(String(symmetry))), nothing)
    class === nothing && throw(ArgumentError("unknown Altland-Zirnbauer symmetry class"))
    status = GC.@preserve real_values imaginary_values output ccall(
        (:thouless_random_gaussian_matrix, _library()),
        Cint,
        (
            Csize_t,
            UInt32,
            Cdouble,
            Ptr{Cdouble},
            Csize_t,
            Ptr{Cdouble},
            Csize_t,
            _C64Mut,
        ),
        dimension,
        class,
        variance,
        pointer(real_values),
        length(real_values),
        pointer(imaginary_values),
        length(imaginary_values),
        _c64_mut(output),
    )
    _check(status)
    return output
end

function _digest(function_name::Symbol, input, salt)
    input_bytes = Vector{UInt8}(codeunits(String(input)))
    salt_bytes = Vector{UInt8}(codeunits(String(salt)))
    output = Ref{Cdouble}()
    status = if function_name === :thouless_digest_uniform
        GC.@preserve input_bytes salt_bytes output ccall(
            (:thouless_digest_uniform, _library()),
            Cint,
            (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, Ref{Cdouble}),
            pointer(input_bytes),
            length(input_bytes),
            pointer(salt_bytes),
            length(salt_bytes),
            output,
        )
    elseif function_name === :thouless_digest_gaussian
        GC.@preserve input_bytes salt_bytes output ccall(
            (:thouless_digest_gaussian, _library()),
            Cint,
            (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, Ref{Cdouble}),
            pointer(input_bytes),
            length(input_bytes),
            pointer(salt_bytes),
            length(salt_bytes),
            output,
        )
    else
        throw(ArgumentError("unknown digest function"))
    end
    _check(status)
    return output[]
end

function _lll_reduce(basis; reduction_parameter=1.34)
    matrix = _real_matrix(basis)
    reduced = similar(matrix)
    raw_transform = Vector{Int64}(undef, length(matrix))
    status = GC.@preserve matrix reduced raw_transform ccall(
        (:thouless_lll_reduce, _library()),
        Cint,
        (_F64View, Cdouble, _F64Mut, Ptr{Int64}, Csize_t),
        _f64_view(matrix),
        reduction_parameter,
        _f64_mut(reduced),
        pointer(raw_transform),
        length(raw_transform),
    )
    _check(status)
    transform = permutedims(reshape(raw_transform, size(matrix, 2), size(matrix, 1)))
    return (basis=reduced, transformation=transform)
end

function _compress_graph(node_count, tails, heads)
    tail_values = Int64.(collect(tails) .- 1)
    head_values = Int64.(collect(heads) .- 1)
    length(tail_values) == length(head_values) ||
        throw(ArgumentError("tails and heads must have the same length"))
    offsets = Vector{Csize_t}(undef, node_count + 1)
    neighbors = Vector{Int64}(undef, length(tail_values))
    status = GC.@preserve tail_values head_values offsets neighbors ccall(
        (:thouless_graph_compress, _library()),
        Cint,
        (
            Csize_t,
            Ptr{Int64},
            Ptr{Int64},
            Csize_t,
            Ptr{Csize_t},
            Csize_t,
            Ptr{Int64},
            Csize_t,
        ),
        node_count,
        pointer(tail_values),
        pointer(head_values),
        length(tail_values),
        pointer(offsets),
        length(offsets),
        pointer(neighbors),
        length(neighbors),
    )
    _check(status)
    return (row_offsets=Int.(offsets) .+ 1, neighbors=Int.(neighbors) .+ 1)
end

function _dense_schur(value)
    matrix = _complex_matrix(value)
    size(matrix, 1) == size(matrix, 2) || throw(ArgumentError("matrix must be square"))
    count = size(matrix, 1)
    form = similar(matrix)
    vectors = similar(matrix)
    eigenvalues = Vector{_C64}(undef, count)
    status = GC.@preserve matrix form vectors eigenvalues ccall(
        (:thouless_dense_schur, _library()),
        Cint,
        (_C64View, _C64Mut, _C64Mut, Ptr{_C64}, Csize_t),
        _c64_view(matrix),
        _c64_mut(form),
        _c64_mut(vectors),
        pointer(eigenvalues),
        count,
    )
    _check(status)
    return (form=form, vectors=vectors, eigenvalues=ComplexF64.(eigenvalues))
end

function _sparse_solve(rows, columns, row_offsets, column_indices, values, right_hand_side)
    offsets = Csize_t.(collect(row_offsets) .- 1)
    indices = Csize_t.(collect(column_indices) .- 1)
    coefficients = _C64.(ComplexF64.(collect(values)))
    right = _complex_matrix(right_hand_side)
    output = Matrix{ComplexF64}(undef, rows, size(right, 2))
    status = GC.@preserve offsets indices coefficients right output ccall(
        (:thouless_sparse_solve, _library()),
        Cint,
        (
            Csize_t,
            Csize_t,
            Ptr{Csize_t},
            Csize_t,
            Ptr{Csize_t},
            Ptr{_C64},
            Csize_t,
            _C64View,
            _C64Mut,
        ),
        rows,
        columns,
        pointer(offsets),
        length(offsets),
        pointer(indices),
        pointer(coefficients),
        length(coefficients),
        _c64_view(right),
        _c64_mut(output),
    )
    _check(status)
    return output
end

module Spectrum
import ..Thouless
hermitian_eigensystem(args...; kwargs...) = Thouless._hermitian_eigensystem(args...; kwargs...)
lead_bands(args...; kwargs...) = Thouless._lead_bands(args...; kwargs...)
export hermitian_eigensystem, lead_bands
end

module AD
using ..Thouless: _ad_affine_projector_value_and_grad
export affine_projector_value_and_grad
affine_projector_value_and_grad(args...; kwargs...) =
    _ad_affine_projector_value_and_grad(args...; kwargs...)
end

module KPM
import ..Thouless
rescale(args...; kwargs...) = Thouless._kpm_rescale(args...; kwargs...)
export rescale
end

module Geometry
import ..Thouless
finite_cluster(args...; kwargs...) = Thouless._finite_cluster(args...; kwargs...)
finite_geometry(args...; kwargs...) = Thouless._finite_geometry(args...; kwargs...)
remove_orbitals(args...; kwargs...) = Thouless._remove_orbitals(args...; kwargs...)
bloch_phase(args...; kwargs...) = Thouless._bloch_phase(args...; kwargs...)
reciprocal_path(args...; kwargs...) = Thouless._reciprocal_path(args...; kwargs...)
lll_reduce(args...; kwargs...) = Thouless._lll_reduce(args...; kwargs...)
export finite_cluster, finite_geometry, remove_orbitals, bloch_phase, reciprocal_path, lll_reduce
end

module Visualization
import ..Thouless
interpolate_density(args...; kwargs...) = Thouless._interpolate_density(args...; kwargs...)
export interpolate_density
end

module Continuum
import ..Thouless
momentum_stencil(args...; kwargs...) = Thouless._momentum_stencil(args...; kwargs...)
export momentum_stencil
end

module Topology
import ..Thouless
wilson_phase(args...; kwargs...) = Thouless._wilson_phase(args...; kwargs...)
chern_numbers(args...; kwargs...) = Thouless._chern_numbers(args...; kwargs...)
quantum_geometric_tensor(args...; kwargs...) =
    Thouless._quantum_geometric_tensor(args...; kwargs...)
local_chern_marker(args...; kwargs...) = Thouless._local_chern_marker(args...; kwargs...)
export wilson_phase, chern_numbers, quantum_geometric_tensor, local_chern_marker
end

module Wannier
import ..Thouless
project_trials(args...; kwargs...) = Thouless._project_trials(args...; kwargs...)
export project_trials
end

module Response
import ..Thouless
intrinsic_curvature(args...; kwargs...) = Thouless._intrinsic_curvature(args...; kwargs...)
export intrinsic_curvature
end

module Observables
import ..Thouless
project_diagonal(args...; kwargs...) = Thouless._project_diagonal(args...; kwargs...)
export project_diagonal
end

module Transport
import ..Thouless

struct Lead
    cell_hamiltonian::Matrix{ComplexF64}
    inter_cell_hopping::Matrix{ComplexF64}
    coupling::Matrix{ComplexF64}

    Lead(cell, hopping, coupling) = new(
        Matrix{ComplexF64}(cell),
        Matrix{ComplexF64}(hopping),
        Matrix{ComplexF64}(coupling),
    )
end

transmissions(args...; kwargs...) = Thouless._transmissions(args...; kwargs...)
propagating_modes(args...; kwargs...) = Thouless._lead_modes(args...; kwargs...)
lead_self_energy(args...; kwargs...) = Thouless._lead_self_energy(args...; kwargs...)
export Lead, transmissions, propagating_modes, lead_self_energy
end

module Symmetry
import ..Thouless
validate_chiral(args...; kwargs...) = Thouless._validate_chiral(args...; kwargs...)
particle_hole_basis(args...; kwargs...) = Thouless._particle_hole_basis(args...; kwargs...)
export validate_chiral, particle_hole_basis
end

module Random
import ..Thouless
gaussian_matrix(args...; kwargs...) = Thouless._gaussian_matrix(args...; kwargs...)
uniform(input, salt) = Thouless._digest(:thouless_digest_uniform, input, salt)
gaussian(input, salt) = Thouless._digest(:thouless_digest_gaussian, input, salt)
export gaussian_matrix, uniform, gaussian
end

module Graph
import ..Thouless
compress(args...; kwargs...) = Thouless._compress_graph(args...; kwargs...)
export compress
end

module LinearAlgebra
import ..Thouless
schur(args...; kwargs...) = Thouless._dense_schur(args...; kwargs...)
sparse_solve(args...; kwargs...) = Thouless._sparse_solve(args...; kwargs...)
export schur, sparse_solve
end
