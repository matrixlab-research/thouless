# Thouless.jl

`julia/Thouless` is a Julia-native interface to the versioned Thouless C ABI.
It does not reimplement scientific algorithms or derivatives in Julia. Model construction,
spectra, geometry, topology, Wannier projection, intrinsic response,
observables, transport, symmetry, random ensembles, graphs, and dense and
sparse algebra all execute in the Rust core.

## Loading the native artifact

A release bundle installs the platform library under
`julia/Thouless/deps/usr/lib`. During development and CI, point the package at
the exact built artifact:

```bash
export THOULESS_LIBRARY="$PWD/target/release/libthouless_capi.so"
julia --project=julia/Thouless -e 'using Pkg; Pkg.test()'
```

Use `.dylib` on macOS and `.dll` on Windows. Package loading verifies the ABI
major version before creating a model.

## Julia conventions

- Public arrays are ordinary `Matrix{Float64}`, `Matrix{ComplexF64}`, vectors,
  and named tuples. The ABI consumes Julia's column-major arrays directly by
  carrying element strides.
- Julia indexing is one-based. Orbital, graph, sparse-matrix, occupied-state,
  and periodic-axis indices are converted at the boundary.
- `ModelBuilder` and `Model` own opaque native handles. Safe finalizers release
  them exactly once. A built builder cannot be reused.
- Native status codes become `ThoulessError` with the thread-local Rust
  diagnostic.
- Numerical precision and coordinate conventions are identical to the Rust
  and C contracts.

The workflow modules are `AD`, `Spectrum`, `KPM`, `Geometry`, `Visualization`,
`Continuum`, `Topology`, `Wannier`, `Response`, `Observables`, `Transport`,
`Symmetry`, `Random`, `Graph`, and `LinearAlgebra`.

`AD.affine_projector_value_and_grad` calls the Rust-native projector VJP
through the stable C ABI. See `docs/native-ad.md` for the complex pairing,
validity conditions, benchmarks, and remaining coverage.
