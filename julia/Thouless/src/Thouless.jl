"""
    Thouless

Julia-native model construction and scientific workflows backed by the
Rust-owned Thouless implementation.

Orbital and lattice-axis indices are one-based at this boundary. Domain
operations are grouped into exported modules such as [`Topology`](@ref),
[`Transport`](@ref), and [`AD`](@ref).
"""
module Thouless

include("_abi.jl")
include("model.jl")
include("workflows.jl")

export ThoulessError, abi_version
export Lattice, ModelBuilder, Model, add_orbital!, set_onsite!, add_hopping!
export build, state_count, hamiltonian, eigensystem
export AD, Spectrum, KPM, Geometry, Visualization, Continuum, Topology, Wannier
export Response, Observables, Transport, Symmetry, Random, Graph, LinearAlgebra

end
