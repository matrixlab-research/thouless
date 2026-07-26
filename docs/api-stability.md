# API Stability

Thouless 0.1 defines a supported Rust contract in
`spec/api/thouless-native.toml`. The inventory is organized by scientific
workflow. Public implementation helpers that are absent from the inventory are
not covered by the 0.1 stability promise.

## Compatibility

Within one `0.x` minor contract, a stable item will not be removed, renamed, or
changed in a way that alters its accepted inputs, returned scientific
quantities, coordinate convention, ownership, or error category. A breaking
change requires an explicit contract-version change. CI compares the inventory
with the pull-request base and rejects a stable-contract change that retains
the old version.

Additive functions, types, methods, and error variants are allowed. Stable
error enums are non-exhaustive, so callers must retain a fallback branch.

## Numbers and coordinates

- Stable real precision is IEEE 754 binary64 (`f64`).
- Stable complex precision is complex binary64 (`Complex64`).
- Matrices are owned, finite, row-major values at the public storage boundary.
- Lattice primitive vectors are Cartesian row vectors.
- Orbital positions and Bloch momenta are reduced coordinates unless an API
  explicitly says `cartesian`.
- A hopping matrix maps its source orbital subspace into its target subspace.
- Energies and lengths carry caller-selected units; Thouless never inserts an
  implicit unit conversion.

## Ownership and scale

Scientific objects own their structural data. Read-only accessors return
borrows tied to the owner; computations return newly owned results. Dense
matrices are intended for operations whose documented result is dense.
`LinearOperator`, canonical CSR matrices, KPM, and sparse open-system APIs are
the supported paths when dense materialization would be asymptotically
incorrect.

## Errors

Rust uses `Result` and non-exhaustive domain errors. Stable callers may
distinguish:

- invalid scientific input or geometry;
- shape or dtype mismatch;
- numerical convergence or singularity;
- an unsupported backend or feature;
- resource exhaustion;
- an internal invariant violation.

The concrete domain error retains diagnostic detail. Panics indicate a
programming defect rather than a recoverable scientific result.

## Concurrency

Immutable models, matrices, and result objects may be shared according to
their Rust auto-traits. Builders are exclusively mutable. Thouless does not
create a global runtime or silently change the caller's thread-pool size.

## Changing the contract

Edit the inventory and its `contract_version` in the same change, update the
language mapping and direct tests, and describe the migration in the pull
request. `tools/check_native_api.py --base-ref <commit>` enforces this rule.
