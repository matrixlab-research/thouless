using Test
import LinearAlgebra as LA
using Thouless

function chain_model(; onsite=0.0)
    builder = ModelBuilder(Lattice([1.0;;], [1]))
    orbital = add_orbital!(builder, "s", [0.0])
    set_onsite!(builder, orbital, onsite)
    add_hopping!(builder, orbital, orbital, [1], -1.0)
    return build(builder)
end

@testset "model and spectrum" begin
    model = chain_model(onsite=0.25)
    @test state_count(model) == 1
    @test hamiltonian(model, [0.0]) ≈ [-1.75 + 0.0im;;]
    @test eigensystem(model, [0.5]).values ≈ [2.25]
    @test_throws ThoulessError hamiltonian(model, [0.0, 0.5])

    dense = Spectrum.hermitian_eigensystem([1.0 1.0im; -1.0im 2.0])
    @test dense.vectors * LA.Diagonal(dense.values) * dense.vectors' ≈
          ComplexF64[1.0 1.0im; -1.0im 2.0]
    bands = Spectrum.lead_bands([0.0;;], [-1.0;;], 0.25; derivative_order=2)
    @test bands.energies ≈ [-2cos(0.25)]
    @test bands.first_derivatives ≈ [2sin(0.25)]

    rescaled = KPM.rescale(LA.Diagonal(ComplexF64[-1, 1]) |> Matrix)
    @test maximum(abs, LA.eigvals(rescaled.matrix)) < 1

    for _ in 1:100
        temporary = chain_model()
        @test state_count(temporary) == 1
    end
    GC.gc()
end

@testset "geometry, continuum, and visualization" begin
    model = chain_model(onsite=0.25)
    finite = Geometry.finite_cluster(model, reshape(Cint[0, 1, 2], 3, 1))
    @test state_count(finite) == 3
    @test sort(real.(LA.eigvals(hamiltonian(finite)))) ≈
          0.25 .+ [-sqrt(2.0), 0.0, sqrt(2.0)]
    vacancy = Geometry.finite_geometry(
        model,
        reshape(Cint[0, 2], 2, 1),
        [1, 1],
    )
    @test hamiltonian(vacancy) ≈ 0.25 .* Matrix{ComplexF64}(LA.I, 2, 2)
    @test Geometry.bloch_phase([1], [0.25]) ≈ cis(0.25)

    path = Geometry.reciprocal_path(
        Lattice([2.0 0.0; 0.0 1.0], [1, 2]),
        [0.0 0.0; 0.5 0.0; 0.5 0.5],
        7,
    )
    @test size(path.points) == (7, 2)
    @test path.node_distances[end] ≈ 1.5pi
    reduced = Geometry.lll_reduce([1.0 1.0; 0.0 1.0])
    @test reduced.transformation * [1.0 1.0; 0.0 1.0] ≈ reduced.basis

    stencil = Continuum.momentum_stencil(1, [(1, 2)])
    @test sort(vec(stencil.offsets)) == [-1, 0, 0, 1]
    @test sum(stencil.weights) ≈ 0

    field = Visualization.interpolate_density(
        [0.0 0.0; 1.0 0.0],
        [1.0, 2.0],
        [0.0 0.0],
        [1.0 0.0];
        absolute_width=0.2,
    )
    @test prod(field.shape) == length(field.values)
    @test size(field.bounds) == (2, 2)
end

@testset "topology, Wannier, response, and observables" begin
    frames = [ComplexF64[1;;], ComplexF64[im;;], ComplexF64[1;;]]
    @test Topology.wilson_phase(frames) ≈ 0 atol = 1.0e-12

    hamiltonian_matrix = ComplexF64[-1 0; 0 1]
    derivatives = [ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0]]
    tensor = Topology.quantum_geometric_tensor(hamiltonian_matrix, derivatives, [1])
    @test size(tensor) == (1, 1, 2, 2)
    @test isfinite(real(tensor[1, 1, 1, 1]))

    marker = Topology.local_chern_marker(
        hamiltonian_matrix,
        [0.0 0.0; 1.0 0.0],
        [1],
        1.0,
    )
    @test length(marker) == 2
    projected = Wannier.project_trials(
        [Matrix{ComplexF64}(LA.I, 2, 2)],
        [1.0 0.0],
    )
    @test length(projected) == 1
    @test projected[1] * projected[1]' ≈ [1.0;;]

    curvature = Response.intrinsic_curvature(chain_model(), [0.1])
    @test curvature ≈ zeros(1, 1)
    observable = Observables.project_diagonal(
        Matrix{ComplexF64}(LA.I, 2, 2),
        [1.0, 2.0],
    )
    @test observable ≈ ComplexF64[1 0; 0 2]
end

@testset "transport and symmetry" begin
    cell = ComplexF64[0;;]
    hopping = ComplexF64[-1;;]
    self_energy = Transport.lead_self_energy(cell, hopping)
    @test self_energy[1, 1] ≈ -im atol = 1.0e-5
    modes = Transport.propagating_modes(cell, hopping)
    @test modes.incoming_count == 1
    @test sort(modes.velocities) ≈ [-2.0, 2.0]
    lead = Transport.Lead(cell, hopping, hopping)
    transmissions = Transport.transmissions(cell, [lead, lead], 0.0)
    @test transmissions[1, 2] ≈ 1.0 atol = 2.0e-6

    @test Symmetry.validate_chiral(
        ComplexF64[0 1; 1 0],
        ComplexF64[1 0; 0 -1],
    ) == 0
    adapted = Symmetry.particle_hole_basis(
        Matrix{ComplexF64}(LA.I, 2, 2),
        Matrix{ComplexF64}(LA.I, 2, 2),
    )
    @test adapted.wave_functions' * adapted.wave_functions ≈
          Matrix{ComplexF64}(LA.I, 2, 2)
    @test adapted.ordering == [1, 1]
end

@testset "random, graph, and linear algebra" begin
    @test 0 <= Random.uniform("model", "salt") < 1
    @test isfinite(Random.gaussian("model", "salt"))
    random_matrix = Random.gaussian_matrix(
        2,
        :A,
        1.0,
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    )
    @test random_matrix ≈ random_matrix'

    graph = Graph.compress(3, [1, 2], [2, 3])
    @test graph.row_offsets == [1, 2, 3, 3]
    @test graph.neighbors == [2, 3]

    matrix = ComplexF64[1 2im; 0 3]
    decomposition = Thouless.LinearAlgebra.schur(matrix)
    @test decomposition.vectors * decomposition.form * decomposition.vectors' ≈ matrix
    solution = Thouless.LinearAlgebra.sparse_solve(
        2,
        2,
        [1, 3, 5],
        [1, 2, 1, 2],
        ComplexF64[2, im, -im, 3],
        ComplexF64[1; 2;;],
    )
    @test ComplexF64[2 im; -im 3] * solution ≈ ComplexF64[1; 2;;]
end
