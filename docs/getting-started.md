# Getting started

Thouless currently builds from source. Rust 1.85 or newer, Python 3.12 or newer,
and Julia 1.10 or newer are the minimum supported language versions. A Fortran
linker is required by the LAPACK-backed native build.

## Rust

Add the repository package to a Cargo workspace while the first release is
being prepared:

```toml
[dependencies]
thouless = { git = "https://github.com/matrixlab-research/thouless" }
```

The native model builder is the central entry point. See the
[generated Rust reference](https://matrixlab-research.github.io/thouless/rust/thouless/)
for modules and types.

## Python

Build and install the wheel from a checkout:

```bash
python -m pip install maturin==1.14.1
maturin build --release --out dist
python -m pip install dist/thouless-*.whl
```

Python arrays are converted at the binding boundary and results are returned
as NumPy arrays. See the
[generated Python reference](https://matrixlab-research.github.io/thouless/python/).

## Julia

Build the C ABI artifact, install it into the Julia package artifact directory,
and activate the package:

```bash
cargo build --release -p thouless-capi
python tools/install_julia_library.py --profile release
julia --project=julia/Thouless -e 'using Pkg; Pkg.instantiate(); using Thouless'
```

Julia uses one-based orbital and axis indices. See the
[generated Julia reference](https://matrixlab-research.github.io/thouless/julia/).

## Reproduce the checks

The commands below exercise the published package boundaries rather than only
internal kernels:

```bash
cargo test --workspace --all-features
python tools/test_built_wheel.py
julia --startup-file=no --project=julia/Thouless -e 'using Pkg; Pkg.test()'
python tools/check_contracts.py
```

The complete CI definition also runs cross-language conformance and pinned
source-compatibility suites.
