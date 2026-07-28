# Thouless

Thouless is a Rust-native scientific package for tight-binding models,
topological observables, and steady-state quantum transport. Rust owns the
scientific implementation. Python and Julia expose first-class interfaces to
the same kernels rather than maintaining independent numerical
implementations.

The public surface is organized around reusable scientific capabilities:

- immutable lattice models with explicit orbital and periodic-cell structure;
- dense and sparse spectral algorithms;
- reciprocal-space geometry, topology, response, and Wannier workflows;
- local observables, finite geometry, and lead-based transport;
- structure-preserving transformations and native automatic differentiation.

## Choose an interface

| Interface | Best for | Generated reference |
| --- | --- | --- |
| Rust | native applications and reusable low-level composition | [Rust API](https://matrixlab-research.github.io/thouless/rust/thouless/) |
| Python | NumPy/SciPy workflows and interactive analysis | [Python API](https://matrixlab-research.github.io/thouless/python/) |
| Julia | Julia-native model construction and numerical workflows | [Julia API](https://matrixlab-research.github.io/thouless/julia/) |
| C ABI | stable foreign-function integration | [C ABI guide](c-api.md) |

All three high-level interfaces are checked against the same semantic
conformance cases. The [native language design](native-language-api-design.md)
describes the ownership boundary and the versioned capability map.

## Project status

The native scientific implementation, Rust/Python/Julia interfaces, stable C
ABI, and pinned PythTB/Kwant compatibility suites are implemented. Independent
held-out validation remains open and is tracked in the repository issues and
coverage matrices; it is not represented as complete by public CI alone.

Start with [Getting started](getting-started.md), then use the generated
reference for the language in which you are composing a workflow.
