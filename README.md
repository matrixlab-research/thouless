# Thouless

[![CI](https://github.com/matrixlab-research/thouless/actions/workflows/ci.yml/badge.svg)](https://github.com/matrixlab-research/thouless/actions/workflows/ci.yml)
[![Documentation](https://github.com/matrixlab-research/thouless/actions/workflows/docs.yml/badge.svg)](https://matrixlab-research.github.io/thouless/)
[![Release](https://github.com/matrixlab-research/thouless/actions/workflows/release.yml/badge.svg)](https://github.com/matrixlab-research/thouless/releases)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-dea584.svg?logo=rust)](https://www.rust-lang.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Julia 1.10+](https://img.shields.io/badge/julia-1.10%2B-9558B2.svg?logo=julia&logoColor=white)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)

Rust-native tight-binding, topology, and steady-state quantum transport, with
first-class Rust, Python, and Julia interfaces.

Thouless uses one scientific implementation across periodic bulk calculations,
finite boundaries, and open systems. The API is organized around physical
objects and reusable numerical capabilities. Python and Julia call the same
Rust kernels, while optional compatibility layers reproduce the pinned
PythTB 2.0 and Kwant 1.5 interfaces.

[Documentation](https://matrixlab-research.github.io/thouless/) ·
[Rust API](https://matrixlab-research.github.io/thouless/rust/thouless/) ·
[Python API](https://matrixlab-research.github.io/thouless/python/) ·
[Julia API](https://matrixlab-research.github.io/thouless/julia/) ·
[Issues](https://github.com/matrixlab-research/thouless/issues)

## What is implemented

- immutable tight-binding models, Bloch Hamiltonians, eigensystems, analytic
  momentum derivatives, supercells, arbitrary finite geometries, and vacancies;
- reciprocal paths and meshes, Wilson loops, first and second Chern invariants,
  quantum geometry, local Chern markers, and intrinsic Berry response;
- local densities, bond currents, source terms, KPM spectra and response,
  Wannier projection, localization, and interpolation;
- periodic lead modes, surface self energies, Green functions, transmissions,
  scattering states, LDOS, and shot noise;
- dense Schur and generalized Schur decompositions, sparse direct solves,
  lattice reduction, compressed graphs, discrete symmetries, and all ten
  Altland-Zirnbauer random-matrix classes;
- Rust-native first-order automatic differentiation for affine Hamiltonians,
  isolated eigensystems, occupied projectors, linear solves, surface Green
  functions, transport objectives, sparse GMRES, and checkpointed KPM.

The Rust, Python, and Julia surfaces map the same 27 stable scientific workflows
in
[`spec/api/thouless-native-languages.toml`](spec/api/thouless-native-languages.toml).
The compatibility build passes all 98 pinned PythTB 2.0 tests and all 398
pinned Kwant 1.5 tests without relaxing their tolerances.

**Project status:** the native implementation, three first-class language
interfaces, stable C ABI, and pinned source-compatibility suites are implemented.
Evaluator-owned isolated held-out validation remains pending in
[issue #6](https://github.com/matrixlab-research/thouless/issues/6). Public CI
therefore establishes executable public evidence, not independent completion
of the held-out gate.

## Design

```text
Rust application ------------------------------+
                                                |
Python application -> typed PyO3 interface -----+-> Rust scientific core
                                                |
Julia application -> versioned C ABI -----------+
                                                |
PythTB/Kwant code -> compatibility adapters ----+
```

The compatibility adapters convert values, preserve source-language state, and
translate errors. They do not contain independent scientific algorithms or
recognize test fixtures. The architectural contract is documented in
[`docs/native-language-api-design.md`](docs/native-language-api-design.md);
native AD semantics and current boundaries are documented in
[`docs/native-ad.md`](docs/native-ad.md).

## Quick start

Thouless currently builds from source. The minimum supported versions are Rust
1.85, Python 3.12, and Julia 1.10. The LAPACK-backed build also needs a Fortran
linker.

### Rust

```toml
[dependencies]
thouless = { git = "https://github.com/matrixlab-research/thouless" }
```

```rust
use thouless::model::{Lattice, ModelBuilder};
use thouless::Complex64;

let lattice = Lattice::new(vec![vec![1.0]], vec![0])?;
let mut builder = ModelBuilder::new(lattice);
let orbital = builder.add_orbital("s", [0.0])?;
builder.set_onsite(orbital, 0.25)?;
builder.add_hopping(
    orbital,
    orbital,
    [1],
    Complex64::new(-1.0, 0.0),
)?;
let model = builder.build()?;
let spectrum = model.eigensystem(&[0.25])?;

# Ok::<(), Box<dyn std::error::Error>>(())
```

### Python

```bash
python -m pip install maturin==1.14.1
maturin build --release --out dist
python -m pip install dist/thouless-*.whl
```

```python
import thouless

lattice = thouless.Lattice([[1.0]], [0])
builder = thouless.ModelBuilder(lattice)
orbital = builder.add_orbital("s", [0.0])
builder.set_onsite(orbital, 0.25)
builder.add_hopping(orbital, orbital, [1], -1.0)
model = builder.build()
energies = model.eigensystem([0.25]).eigenvalues
```

### Julia

```bash
cargo build --release -p thouless-capi
python tools/install_julia_library.py --profile release
julia --project=julia/Thouless
```

```julia
using Thouless

lattice = Lattice(reshape([1.0], 1, 1), [1])
builder = ModelBuilder(lattice)
orbital = add_orbital!(builder, "s", [0.0])
set_onsite!(builder, orbital, 0.25)
add_hopping!(builder, orbital, orbital, [1], -1.0)
model = build(builder)
energies = eigensystem(model, [0.25]).values
```

See the [getting-started guide](docs/getting-started.md) and generated
language references for complete signatures, shapes, indexing conventions, and
failure modes.

## Documentation

Documentation is generated from the public code contracts:

- Rust uses rustdoc with missing public documentation denied by the crate;
- Python uses Sphinx autodoc and an AST gate over every local `__all__` symbol
  and public method;
- Julia uses Documenter.jl with `checkdocs=:exports`.

Build the combined portal after installing `docs/requirements.txt`, the Python
wheel, the Julia native library, and the Julia docs environment:

```bash
python tools/check_python_docs.py
julia --project=julia/Thouless/docs -e 'using Pkg; Pkg.instantiate()'
python tools/build_docs.py
```

The result is written to `target/site`; CI publishes the same tree to GitHub
Pages.

## Verification

The repository keeps four distinct kinds of evidence:

- `tests/` directly exercises the Rust-native API;
- `python-tests/` and `julia/Thouless/test/` exercise installed language
  packages and cross-language semantic conformance;
- `compat-tests/` and pinned upstream suites exercise source-compatible PythTB
  and Kwant entry points;
- `spec/coverage/` and `spec/api/` connect capabilities and public interfaces
  to tests, implementation state, and issues.

Run the core local checks with:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --all-features --no-deps
python tools/check_contracts.py
python tools/check_native_api.py
python tools/check_capi_julia.py
python tools/check_python_docs.py
maturin build --release --out dist
python tools/test_built_wheel.py
cargo build --release -p thouless-capi
python tools/run_c_smoke.py --profile release
python tools/install_julia_library.py --profile release
julia --startup-file=no --project=julia/Thouless -e 'using Pkg; Pkg.test()'
```

CI additionally verifies cross-language analytic invariants, realistic
scientific fixtures, scale guards against dense materialization, AD
correctness/performance contracts, and the complete pinned source suites. Any
intentional source-test skip must link to an open issue.

## Compatibility and provenance

The source baselines are
[PythTB 2.0.0](https://pythtb.readthedocs.io/en/stable/) and
[Kwant 1.5.0](https://kwant-project.org/doc/latest/). No source implementation
or upstream test has been copied into this repository. Vendored reference
fixtures retain exact provenance and their original licenses.

Completed compatibility work remains auditable in
[issue #4](https://github.com/matrixlab-research/thouless/issues/4) and
[issue #5](https://github.com/matrixlab-research/thouless/issues/5). Native
language and AD design decisions are recorded in
[issues #7–#10](https://github.com/matrixlab-research/thouless/issues/7) and
[issue #14](https://github.com/matrixlab-research/thouless/issues/14).

## Contributing

Coding agents and human contributors should begin with
[`AGENTS.md`](AGENTS.md). It points to the repository's complete
first-principles reimplementation instruction, compatibility evidence policy,
anti-overfitting constraints, issue audit trail, and status semantics.

## License

Thouless is licensed under the [MIT License](LICENSE-MIT). Upstream fixtures
retain their own licenses as listed in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
