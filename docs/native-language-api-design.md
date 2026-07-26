# Native Rust, Python, and Julia API Design

## Status

This document defines the implemented language-native API contract for
Thouless. The versioned inventories and coverage matrices are executable
acceptance evidence.

- the Rust core implements the public scientific capabilities recorded in
  `spec/coverage/native.toml`, except isolated held-out validation;
- the installed Python wheel exposes a typed supported API while retaining
  `thouless._core` as a private implementation bridge;
- `thouless-capi` provides a generated versioned C header and `Thouless.jl`
  exposes the same 26 workflows;
- pull-request CI installs produced artifacts, exercises Rust/Python/Julia
  parity, and retains all pinned source-package tests;
- release CI builds install-tested wheels and platform Julia/C bundles;
- scheduled CI records ownership, scientific-scale, performance, and
  allocation evidence.

The project therefore remains `Incomplete`.

## Decision

Thouless will provide one stable scientific contract with three language
surfaces:

1. the idiomatic Rust-native API in the `thouless` crate;
2. a first-class Python-native API backed directly by PyO3;
3. a first-class Julia-native API backed by a stable C ABI.

The three surfaces must expose the same stable scientific workflows and
semantics. They do not need identical syntax, identical object ownership, or a
one-to-one wrapper for every Rust helper function.

The PythTB and Kwant compatibility layers remain separate source-compatible
adapters. Their historical APIs do not define the Rust, Python-native, or
Julia-native architecture.

## Target Architecture

```text
                         +-> Python-native `thouless` API
                         |
Rust `thouless` core -> PyO3 extension `thouless._core`
          |              |
          |              +-> PythTB and Kwant compatibility adapters
          |
          +-> versioned `thouless-capi` -> Julia-native `Thouless.jl`
```

`spec/api/thouless-native-languages.toml` is the machine-readable mapping
between scientific capability identifiers and language namespaces. CI must
reject a Rust-native capability that is absent from this design contract.

No additional generic binding framework is required. Python and Julia have
different runtime strengths:

- PyO3 provides natural Python exceptions, NumPy integration, Python object
  lifetimes, and GIL management.
- A small C ABI provides Julia with a stable `ccall` boundary without imposing
  Python or C++ as an intermediate runtime.

Both adapters call the same reusable Rust core. Neither adapter may contain an
independent scientific algorithm.

## Stable Contract Boundary

The stable contract is defined in units of scientific workflows, not in units
of implementation functions. Every capability in
`spec/coverage/native.toml`, other than evaluator-owned validation, must map to
a reachable Rust, Python, and Julia namespace.

The stable surface should be centered on these objects and operations:

| Area | Stable concepts |
| --- | --- |
| Model | lattice, orbital, hopping, model builder, immutable model |
| Geometry | finite selections, transformations, reciprocal paths and meshes |
| Spectrum | Hamiltonians, eigensystems, bands, sparse operators, KPM |
| Topology | connections, Wilson loops, Chern and Z2 invariants, quantum geometry |
| Wannier | projection, disentanglement, localization, interpolation |
| Response | intrinsic Hall geometry and Berry-curvature dipoles |
| Observables | density, current, source, and local operator evaluation |
| Transport | leads, self-energies, modes, open systems, scattering results |
| Supporting mathematics | symmetry, decomposition, lattice reduction, graph and random ensembles |

An internal Rust helper does not become stable merely because it is currently
`pub`. Conversely, one high-level operation may compose several Rust
capabilities. The stable inventory must be explicit before the crate advances
from `0.0.x` to a supported `0.1` contract.

### Rust requirements

- Preserve idiomatic ownership and `Result`-based errors.
- Document coordinate conventions, units, precision, sparse/dense behavior,
  and algorithmic scale boundaries.
- Give every stable workflow a direct Rust test or executable example.
- Maintain a machine-checkable stable API inventory.
- Treat removal or semantic change of a stable entry as a contract version
  change.

### Python requirements

The public package must expose documented modules such as:

```text
thouless.model
thouless.geometry
thouless.spectrum
thouless.topology
thouless.wannier
thouless.response
thouless.observables
thouless.transport
thouless.linalg
```

`thouless._core` remains an implementation detail. Users must not need to
import it directly.

Python objects should own or share Rust values through PyO3. Public functions
must:

- accept and return documented NumPy-compatible shapes and dtypes;
- translate Rust errors into stable Python exception classes;
- release the GIL around long Rust computations when no Python callback is
  active;
- expose Python callback support only where a real scientific workflow
  requires it;
- provide type information for the supported public surface;
- avoid reimplementing model assembly, solvers, topology, response, or
  transport in Python.

The compatibility packages may continue to call private extension operations
where their source semantics require a lower-level boundary. Those operations
do not automatically become part of the Python-native API.

### Julia requirements

Add:

- `crates/thouless-capi`, containing the public C ABI;
- a generated and versioned C header;
- `julia/Thouless`, containing the Julia package and tests.

The C ABI should use opaque owned handles for long-lived Rust objects and flat
descriptors for arrays and scalar data. It must define:

- an ABI version query;
- explicit creation and destruction for every owned handle;
- panic containment at every exported entry point;
- stable status codes and retrievable UTF-8 error messages;
- C-compatible complex values;
- shape, stride, element type, mutability, and ownership for every array;
- rules for borrowed inputs and owned outputs;
- thread-safety and concurrency guarantees.

The initial ABI should copy result arrays into language-owned memory unless a
measured workflow requires zero-copy behavior. Correct ownership is more
important than speculative zero-copy complexity.

`Thouless.jl` should wrap handles in Julia types with safe finalizers and
translate status codes into Julia exceptions. Julia users should see ordinary
Julia arrays and scientific objects, not raw pointers.

## Shared Semantic Rules

### Numbers and arrays

- Stable scalar precision is `f64` and complex precision is complex `f64`
  unless a capability explicitly documents another type.
- Shapes and axis meanings are part of the public contract.
- Rust row-major storage, NumPy layout, and Julia column-major layout must be
  converted explicitly. A binding must not infer layout from shape alone.
- Inputs may accept compatible contiguous layouts when the adapter can prove
  the interpretation. Outputs should use the receiving language's normal
  layout.

### Errors

The stable error taxonomy should distinguish at least:

- invalid scientific input or geometry;
- shape or dtype mismatch;
- numerical convergence or singularity;
- unsupported backend or feature;
- resource exhaustion;
- internal invariant violation.

Error messages may add detail, but callers must be able to branch on a stable
error class or status code.

### Lifetimes and callbacks

- Rust remains the source of truth for object validity.
- Python and Julia wrappers must not expose a use-after-free path.
- Callback lifetimes and thread entry rules must be documented separately from
  ordinary data-only calls.
- Declarative onsite, hopping, and operator data are preferred over callbacks
  when both express the same workflow.

### Concurrency

- Long-running data-only Python operations should release the GIL.
- Julia calls must document whether they are thread-safe and whether the Julia
  garbage collector may run during the native call.
- Rust internal parallelism must not silently oversubscribe host-language
  thread pools.

## Coverage and Issue Rules

Once implementation starts, add:

- `spec/coverage/python-native.toml`;
- `spec/coverage/julia-native.toml`.

Each matrix must use the same capability identifiers as
`spec/coverage/native.toml`. A row is `implemented` only when the public
language API, native-language tests, documentation, and installable artifact
all exist. Every other row must link to an open reproducible GitHub issue.

The Rust stable inventory, Python inventory, Julia inventory, and the three
capability matrices answer different questions:

- the inventory records names and signatures;
- the capability matrix records reachable scientific behavior;
- source-compatibility matrices record PythTB and Kwant compatibility;
- held-out evaluation records independent generality evidence.

None substitutes for another.

## Required CI

CI must validate installed artifacts and scientific semantics, not only source
tree imports.

### Required pull-request jobs

`rust-core`

- run formatting, clippy, workspace tests, and rustdoc;
- run on Linux, macOS, and Windows where supported;
- validate the stable Rust API inventory and the language mapping manifest.

`python-native`

- build a wheel with Maturin;
- install that wheel into a clean virtual environment;
- run the public Python-native test suite without adding the repository Python
  source directory to `PYTHONPATH`;
- test every supported Python version on Linux and at least one supported
  version on macOS and Windows;
- validate exceptions, shapes, dtypes, ownership, and GIL behavior.

`julia-native`

- build the exact `thouless-capi` library used by the test;
- install the Julia package against that artifact;
- run `Pkg.test()` on Julia LTS and the current stable Julia release;
- cover Linux, macOS, and Windows;
- validate ABI version, error propagation, array conversion, finalization, and
  repeated create/destroy behavior.

`language-parity`

- execute the same generated scientific cases through Rust, Python, and Julia;
- include periodic bands, finite geometry, a topological invariant, Wannier or
  response behavior, local observables, and steady-state transport;
- compare physical invariants and operation-specific tolerances;
- never embed evaluator-owned held-out inputs.

`source-compatibility`

- retain all pinned PythTB 2.0 and Kwant 1.5 tests through the Rust-backed
  compatibility packages;
- run the complete source suites on one canonical platform rather than
  multiplying them across every language and operating-system combination.

### Release jobs

- build supported Python wheels and install each produced wheel;
- build and install the Julia native artifact used by `Thouless.jl`;
- run clean-environment import and end-to-end smoke tests;
- record artifact checksums, ABI version, and language package versions.

### Scheduled jobs

- sanitizers and native ownership/leak checks;
- realistic scientific-size workflows;
- performance and allocation regression tracking;
- optional backend coverage.

Scheduled and release jobs do not weaken required pull-request correctness
checks. Held-out tests remain outside the public repository and public CI.

## Shared Conformance Cases

The public parity suite should generate inputs at test time and cover at least:

- an SSH-like one-dimensional model with analytic bands and polarization;
- a Haldane- or Qi-Wu-Zhang-like model with a quantized Chern invariant;
- a finite system with a vacancy and local observables;
- a clean lead-device-lead system with ballistic transmission;
- a composite-band gauge transformation preserving Wilson/Wannier results;
- invalid shapes, singular geometry, and solver failure paths.

The suite must compare against analytic properties or independently computed
invariants where possible. It must not become a table of stored outputs for a
small set of known fixtures.

## Work Items and Acceptance

This is one complete objective. The ordering below expresses technical
dependencies, not separate completion claims:

1. stabilize the Rust-native public contract and executable inventory;
2. expose the first-class Python-native API;
3. define the versioned C ABI and implement `Thouless.jl`;
4. add native-language coverage matrices and all required CI jobs;
5. preserve the existing PythTB and Kwant compatibility evidence;
6. complete independent held-out validation.

The language-native work is complete only when:

- every in-scope scientific capability is reachable through all three stable
  language surfaces or is linked to an open issue;
- the published documentation describes real user workflows;
- clean installations of produced artifacts pass;
- cross-language semantic tests pass without relaxed tolerances;
- source compatibility remains green;
- realistic-scale and ownership checks pass;
- isolated held-out validation passes;
- all associated gap issues are closed.

Implementation and validation gaps are tracked in:

- [issue #7](https://github.com/matrixlab-research/thouless/issues/7):
  stabilize the Rust-native public API contract;
- [issue #8](https://github.com/matrixlab-research/thouless/issues/8):
  provide the first-class Python-native API;
- [issue #9](https://github.com/matrixlab-research/thouless/issues/9):
  add the stable C ABI and `Thouless.jl`;
- [issue #10](https://github.com/matrixlab-research/thouless/issues/10):
  add artifact and cross-language semantic-parity CI.
