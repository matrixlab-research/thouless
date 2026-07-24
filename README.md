# Thouless

**Rust-native tight-binding, topology, and steady-state quantum transport.**

Thouless is intended to provide one scientific model across periodic bulk
calculations, finite boundaries, and open-system transport. The native Rust API
is designed from the physical objects and invariants. Separate Python
compatibility layers will reproduce the in-scope PythTB 2.0 and Kwant 1.5
interfaces while calling the same Rust core.

## Current status

**Incomplete bootstrap.**

The repository currently implements only model-construction objects and their
structural invariants. It does not yet implement eigensolvers, topology,
response theory, open-system transport, or either compatibility package.

A green CI run currently means:

- the implemented Rust model invariants pass;
- the coverage matrices are internally consistent;
- compatibility tests cannot accidentally run against the original packages;
- every intentionally skipped compatibility suite links to an open issue.

It does **not** mean that the scientific package or compatibility targets are
complete.

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

let lattice = Lattice::new(1, vec![vec![1.0]])?;
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
- The exact upstream source-test manifests still need to be pinned in
  [PythTB issue #4](https://github.com/matrixlab-research/thouless/issues/4)
  and [Kwant issue #5](https://github.com/matrixlab-research/thouless/issues/5).
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
python -m pytest -q -ra compat-tests
```

The compatibility tests are expected to skip until the repository supplies
`python/pythtb` and `python/kwant`. A skip without a linked issue is a CI error.

## Source baselines

- [PythTB 2.0.0](https://pythtb.readthedocs.io/en/stable/)
- [Kwant 1.5.0](https://kwant-project.org/doc/latest/)

No source implementation or source test has been copied into this bootstrap.
Any future vendoring of upstream tests must retain its original license and
provenance.

## License

Thouless is licensed under MIT. Upstream tests and fixtures, if added later,
retain their own licenses.
