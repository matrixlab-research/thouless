#![allow(unused_imports)]

use thouless::ad::{
    AffineHermitianFamily, ModelDirection, ModelGradient, ModelParameters,
    SpectralProjectorObjective,
};
use thouless::bands::{BandEvaluation, PeriodicBands};
use thouless::continuum::{
    finite_difference_stencil, landau_ladder_coefficient, DifferentialFactor,
    DifferentialStencilTerm,
};
use thouless::decomposition::{
    generalized_schur, real_schur, schur, RealSchurDecomposition, SchurDecomposition,
};
use thouless::digest::{gaussian, uniform, uniform_pair};
use thouless::geometry::{ReciprocalPath, UniformReciprocalMesh};
use thouless::graph::{CompressedGraph, CompressionOptions, DirectedEdge, DirectedGraphBuilder};
use thouless::interpolation::{
    interpolate_current, interpolate_density, RegularField, SmoothingOptions,
};
use thouless::kpm::{chebyshev_vectors, reconstruct, rescale_sparse_hamiltonian, SpectralScale};
use thouless::lattice_reduction::{
    closest_lattice_vectors, gram_schmidt, lll_reduce, voronoi_neighbors, ReducedBasis,
};
use thouless::lead_modes::{
    propagating_modes, setup_lead_linear_system, LeadLinearSystem, PropagatingLeadModes,
};
use thouless::linear_operator::{CsrMatrix, LinearOperator};
use thouless::model::{Lattice, ModelBuilder, OrbitalId, TightBindingModel};
use thouless::observables::{
    bond_currents, local_densities, local_sources, project_diagonal_observable, LocalBasisLayout,
    LocalOperatorSet,
};
use thouless::periodic::{bloch_phase, fold_terms, PeriodicTerm};
use thouless::random_matrix::{circular_from_components, gaussian_from_components, SymmetryClass};
use thouless::response::{
    band_response_from_model, berry_curvature_dipole, BandResponsePoint, FermiDistribution,
    UniformMeshBandResponse,
};
use thouless::sparse_direct::{schur_complement, SparseLuAnalysis, SparseLuFactorization};
use thouless::spectrum::Eigensystem;
use thouless::symmetry::{
    particle_hole_symmetric_basis, DiscreteSymmetry, ParticleHoleBasis, SymmetryViolation,
};
use thouless::topology::{
    chern_numbers_on_uniform_grid, local_chern_marker_from_hamiltonian,
    local_chern_marker_from_projector, quantum_geometric_tensor_from_hamiltonian_derivatives,
    second_chern_from_hamiltonian_derivatives, wilson_line_phase, wilson_loop_eigenphases,
    z2_invariant_on_uniform_grid, QuantumGeometricTensor,
};
use thouless::transform::{
    make_finite_cluster, make_finite_geometry, make_supercell, remove_orbitals, FiniteGeometry,
    FiniteSite,
};
use thouless::transport::{
    regularize_retarded_self_energy, retarded_lead_self_energy, solve_open_system,
    square_lattice_self_energy, LeadContact, LocalizedSelfEnergy, ScatteringSolution,
    SparseOpenSystem, SurfaceGreenOptions,
};
use thouless::wannier::{
    disentangle_subspace, interpolate_periodic_matrices, inverse_bloch_transform,
    maximize_localization, periodic_overlaps, project_trials, spread_decomposition,
};
use thouless::Complex64;

#[test]
fn stable_model_and_periodic_spectrum_workflow_is_reachable() {
    let _hamiltonian = TightBindingModel::hamiltonian;
    let _eigensystem = TightBindingModel::eigensystem;
    let _band_structure = TightBindingModel::band_structure;

    let lattice = Lattice::new(vec![vec![1.0]], vec![0]).expect("valid lattice");
    let mut builder = ModelBuilder::new(lattice);
    let orbital = builder.add_orbital("s", [0.0]).expect("valid orbital");
    builder.set_onsite(orbital, 0.25).expect("valid onsite");
    builder
        .add_hopping(orbital, orbital, [1], Complex64::new(-1.0, 0.0))
        .expect("valid hopping");
    let model = builder.build().expect("valid model");

    let gamma = model.eigensystem(&[0.0]).expect("valid eigensystem");
    let zone_edge = model.eigensystem(&[0.5]).expect("valid eigensystem");
    assert!((gamma.eigenvalues()[0] + 1.75).abs() <= 1.0e-12);
    assert!((zone_edge.eigenvalues()[0] - 2.25).abs() <= 1.0e-12);
}

#[test]
fn stable_contract_uses_binary64_numbers() {
    assert_eq!(std::mem::size_of::<f64>(), 8);
    assert_eq!(std::mem::size_of::<Complex64>(), 16);
}
