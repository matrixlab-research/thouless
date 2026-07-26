import LinearAlgebra as LA
using Thouless

function ssh_model()
    builder = ModelBuilder(Lattice([1.0;;], [1]))
    first = add_orbital!(builder, "a", [0.0])
    second = add_orbital!(builder, "b", [0.5])
    add_hopping!(builder, first, second, [0], 0.6)
    add_hopping!(builder, first, second, [1], 1.0)
    return build(builder)
end

function qwz_model()
    sigma_x = ComplexF64[0 1; 1 0]
    sigma_y = ComplexF64[0 -im; im 0]
    sigma_z = ComplexF64[1 0; 0 -1]
    builder = ModelBuilder(Lattice([1.0 0.0; 0.0 1.0], [1, 2]))
    orbital = add_orbital!(builder, "spinor", [0.0, 0.0]; degrees_of_freedom=2)
    set_onsite!(builder, orbital, -sigma_z)
    add_hopping!(builder, orbital, orbital, [1, 0], 0.5sigma_z - 0.5im * sigma_x)
    add_hopping!(builder, orbital, orbital, [0, 1], 0.5sigma_z - 0.5im * sigma_y)
    return build(builder)
end

ssh = ssh_model()
zone_edge = eigensystem(ssh, [0.5]).values
frames = [
    permutedims(eigensystem(ssh, [sample / 65]).vectors[:, 1:1]) for
    sample in 0:64
]
push!(frames, frames[1] .* ComplexF64[1 -1])
polarization = mod(Topology.wilson_phase(frames) / (2pi), 1)

chern = abs(Topology.chern_numbers(qwz_model(), [31, 31], [1, 2], [1])[1])

vacancy = Geometry.finite_geometry(ssh, reshape(Cint[0, 2], 2, 1), [1, 2])
projected = Observables.project_diagonal(
    Matrix{ComplexF64}(LA.I, 2, 2),
    [1.0, 2.0],
)

cell = ComplexF64[0;;]
hopping = ComplexF64[-1;;]
lead = Transport.Lead(cell, hopping, hopping)
transmissions = Transport.transmissions(cell, [lead, lead], 0.0)

inverse_sqrt_two = inv(sqrt(2.0))
wilson_frames = [
    ComplexF64[inverse_sqrt_two inverse_sqrt_two],
    ComplexF64[inverse_sqrt_two im * inverse_sqrt_two],
    ComplexF64[inverse_sqrt_two inverse_sqrt_two],
]
phase = Topology.wilson_phase(wilson_frames)
transformed = copy(wilson_frames)
transformed[2] .*= cis(0.37)
gauge_delta = abs(Topology.wilson_phase(transformed) - phase)

invalid_shape = try
    hamiltonian(ssh, [0.0, 0.5])
    0.0
catch error
    error isa ThoulessError ? 1.0 : rethrow()
end

metrics = Dict(
    "ssh_gap" => zone_edge[2] - zone_edge[1],
    "ssh_polarization" => polarization,
    "chern_absolute" => chern,
    "vacancy_states" => state_count(vacancy),
    "vacancy_observable_trace" => real(sum(LA.diag(projected))),
    "ballistic_transmission" => transmissions[2, 1],
    "wilson_gauge_delta" => gauge_delta,
    "invalid_shape_error" => invalid_shape,
)
for name in sort(collect(keys(metrics)))
    println(name, "=", string(Float64(metrics[name])))
end
