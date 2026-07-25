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
Berry fluxes and uniform-grid Chern numbers, reciprocal paths, parallel
transport, local-observable projection, and structure-preserving model
transformations in Rust. It also provides discrete-symmetry validation and all
ten Altland--Zirnbauer Gaussian and circular random-matrix ensembles through
the native core, together with LLL lattice reduction, closest-vector search,
and Voronoi neighbors. The Python extension and partial PythTB 2.0 layer pass
all 98 pinned upstream tests without changing their tolerances. It does
not yet implement the complete PythTB surface, complete topology and response
theory, complete open-system transport, or the complete Kwant compatibility
surface.

A green CI run currently means:

- the implemented Rust model and spectral invariants pass;
- the coverage matrices are internally consistent;
- compatibility tests cannot accidentally run against the original packages;
- the implemented PythTB smoke contracts execute through the Rust extension;
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
- Exact upstream source and test baselines are pinned in `spec/upstream/`.
  Remaining PythTB and Kwant failures are tracked in
  [issue #4](https://github.com/matrixlab-research/thouless/issues/4) and
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
PYTHONPATH=python python -m pytest -q -ra compat-tests
```

The Kwant ballistic smoke contract executes through the Rust transport core,
and a strict 107-test upstream slice validates the compatibility layer;
remaining Kwant gaps are tracked in issue #5. The strict PythTB runner executes
all 98 collected source tests through the repository-built extension. Broader
PythTB public-interface inventory work remains tracked in issue #4. A skip
without a linked issue is a CI error.

## Source baselines

- [PythTB 2.0.0](https://pythtb.readthedocs.io/en/stable/)
- [Kwant 1.5.0](https://kwant-project.org/doc/latest/)

No source implementation or source test has been copied into this bootstrap.
Any future vendoring of upstream tests must retain its original license and
provenance.

## License

Thouless is licensed under MIT. Upstream tests and fixtures, if added later,
retain their own licenses.
