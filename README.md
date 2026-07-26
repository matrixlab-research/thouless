# Thouless

**Rust-native tight-binding, topology, and steady-state quantum transport.**

Thouless is intended to provide one scientific model across periodic bulk
calculations, finite boundaries, and open-system transport. The native Rust API
is designed from the physical objects and invariants. Separate Python
compatibility layers will reproduce the in-scope PythTB 2.0 and Kwant 1.5
interfaces while calling the same Rust core.

## Current status

**Incomplete implementation.**

The repository currently implements reusable model construction, dense
Hermitian assembly, eigensolvers, momentum derivatives, discrete Wilson phases,
Berry fluxes and uniform-grid Chern numbers, metric-aware reciprocal paths and
uniform meshes with explicit quadrature measures, parallel transport,
hybrid Wannier centers, reduced polarization, mesh-converged time-reversal
`Z2` classification, block-sparse local densities, bond currents, onsite
sources, local-observable projection, and structure-preserving model
transformations in Rust. Arbitrary finite site selections provide open
boundaries, incomplete cells, vacancies, and onsite disorder while preserving
source-site provenance. Kernel-polynomial spectra and Kubo-Bastin responses use
canonical CSR operators, matrix-free Lanczos rescaling, Chebyshev recurrences,
sparse observation operators, and sparsity-preserving velocity commutators;
scientific-scale paths never construct a dense Hamiltonian. It also provides
discrete-symmetry validation and all ten Altland--Zirnbauer Gaussian and
circular random-matrix ensembles through the native core, together with LLL
lattice reduction, closest-vector search, Voronoi neighbors, gauge-covariant
Wannier projection, sampled-frame overlaps, multidimensional FFTs, spread
decomposition, and periodic matrix interpolation, including a native
multidimensional maximal-localization optimizer and fixed-rank SMV
disentanglement with frozen subspaces. The native response core additionally
evaluates band-resolved Kubo Berry curvature,
occupation-weighted Hall geometry, and finite-temperature Berry-curvature
dipoles from explicit Hamiltonian derivatives and quadrature weights. The Python
extension and compatibility layers pass all 98 pinned PythTB 2.0 tests and all
398 pinned Kwant 1.5 tests without changing their tolerances. Executable API
manifests cover the public modules, exports, and core object members of both
source packages, including Wannier, visualization, low-level system, solver,
linear-algebra, and local-operator entry points. Kwant density, current, and
source evaluation now resolves the continuity equation in the shared Rust
core; Python performs only site, callback, and array mapping. Complete
steady-state compatibility entry points likewise delegate embedded-self-energy
solves, Green functions, Caroli and Fisher-Lee transport, scattering states,
channel inference, and LDOS to Rust. Device Hamiltonians and contact-local
self-energies are assembled as canonical CSR operators and solved using
Rust-native zero-fill incomplete-LU right preconditioning with restarted GMRES;
100000-dimensional native and 20000-dimensional compatibility contracts reject
dense device materialization, while the pinned quantum Hall workflow verifies
quantized transport. Real and complex ordinary and generalized Schur paths
preserve source dtypes, real quasi-triangular blocks, conjugate-pair order, and
invariant-subspace reordering. The optional Kwant sparse-direct interface now
uses Rust-native fill-reducing symbolic analysis, reusable complex sparse LU,
multiple-right-hand-side solves, and ordered principal Schur complements;
compatibility statistics expose portable storage counts without claiming
MUMPS-internal measurements. Remaining work comprises broader Wannier90 and
Quantum ESPRESSO fixtures, realistic material-scale Wannier validation,
broader intrinsic response theory, and isolated held-out validation.

A green CI run currently means:

- the implemented Rust model and spectral invariants pass;
- the coverage matrices are internally consistent;
- the PythTB and Kwant public API inventories remain importable;
- compatibility tests cannot accidentally run against the original packages;
- all pinned PythTB and Kwant source tests execute through the compatibility
  layers;
- every intentionally skipped compatibility suite links to an open issue.

It does **not** mean that the scientific package or compatibility targets are
complete.

## Agent instructions

All coding agents must begin with [`AGENTS.md`](AGENTS.md), which points to the
complete repository instruction:
[`instructions/scientific-software-reimplementation.md`](instructions/scientific-software-reimplementation.md).

The instruction defines the first-principles Rust API rule, complete
PythTB/Kwant compatibility objective, GitHub issue audit trail, source-test
policy, anti-overfitting requirements, held-out boundary, and status semantics.
CI verifies that this instruction remains present and reachable from
`AGENTS.md`.

## Architecture

```text
PythTB caller/test -> thin Python compatibility layer --+
                                                       |
Kwant caller/test  -> thin Python compatibility layer --+-> Rust core
                                                       |
Rust caller        -> native Rust API ------------------+
```

The compatibility layers may convert data, map state, and translate errors.
They must not contain separate scientific algorithms.

The first native module is:

```rust
use thouless::model::{Lattice, ModelBuilder};
use thouless::Complex64;

let lattice = Lattice::new(vec![vec![1.0]], vec![0])?;
let mut builder = ModelBuilder::new(lattice);
let orbital = builder.add_orbital("s", [0.0])?;
builder.set_onsite(orbital, 0.25)?;
builder.add_hopping(orbital, orbital, [1], Complex64::new(-1.0, 0.0))?;
let model = builder.build()?;
# Ok::<(), thouless::ModelError>(())
```

## Executable contracts

- `tests/` contains direct tests of the Rust-native API.
- `compat-tests/` contains clean-room smoke contracts for source-compatible
  entry points.
- `spec/coverage/` maps scientific capabilities and source interfaces to tests,
  implementation status, and GitHub issues.
- `spec/api/` contains executable public API inventories for the pinned source
  versions.
- Exact upstream source and test baselines are pinned in `spec/upstream/`.
  Remaining PythTB semantic and validation work is tracked in
  [issue #4](https://github.com/matrixlab-research/thouless/issues/4);
  completed Kwant backend work retains its audit trail in
  [issue #5](https://github.com/matrixlab-research/thouless/issues/5).
- Isolated held-out validation is tracked in
  [issue #6](https://github.com/matrixlab-research/thouless/issues/6).

Source tests are executable compatibility evidence, not the definition of the
native architecture. No implementation may recognize known tests or return
stored fixture results.

## Local checks

```console
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
python tools/check_contracts.py
python -m pip install maturin numpy pytest
maturin develop
python tools/check_python_api.py
PYTHONPATH=python python -m pytest -q -ra compat-tests
```

The strict runners execute all 398 collected Kwant tests and all 98 collected
PythTB tests through the repository-built extension. The API checker separately
validates the pinned public surfaces, so a green source-test run cannot hide a
missing untested export. Deeper PythTB semantic parity, Wannier and file-format
differential fixtures, and scientific-scale validation remain tracked in issue
#4. The completed Kwant sparse-direct backend retains its implementation audit
trail in issue #5. A skip without a linked issue is a CI error.

## Source baselines

- [PythTB 2.0.0](https://pythtb.readthedocs.io/en/stable/)
- [Kwant 1.5.0](https://kwant-project.org/doc/latest/)

No source implementation or source test has been copied into this bootstrap.
Any future vendoring of upstream tests must retain its original license and
provenance.

## License

Thouless is licensed under MIT. Upstream tests and fixtures, if added later,
retain their own licenses.
