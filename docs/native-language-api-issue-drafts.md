# Native Language API Issue Drafts

These drafts preserve the required GitHub issue audit trail while repository
write authorization is being established. Create the issues before adding
missing rows to `spec/coverage/python-native.toml` or
`spec/coverage/julia-native.toml`, replace the placeholders in the design PR
with the resulting issue URLs, and then remove this draft file.

The baseline for every issue is Thouless commit
`0d87773278183ddc7c254438dccbda1face04fb2`. The current green CI evidence is
<https://github.com/matrixlab-research/thouless/actions/runs/30191892275>.

## Stabilize the Rust-native public API contract

Labels: `native-api`

Missing capability: a versioned, machine-checkable public Rust API contract.
The implementation exposes a broad surface, but the crate remains at `0.0.0`
with `publish = false`, and the native coverage matrix tracks scientific
capabilities rather than supported names and stability.

Affected interface and tests:

- Rust callers of `thouless`;
- proposed `spec/api/thouless-native.toml`;
- proposed `tools/check_native_api.py`;
- direct examples or tests for every stable workflow.

Reproduction:

```console
sed -n '1,120p' Cargo.toml
sed -n '1,220p' src/lib.rs
find spec/api -maxdepth 1 -type f -print
```

Expected: a small complete API organized around scientific objects, with a
stability policy and executable inventory.

Actual: public implementation symbols exist without a stable native inventory
or compatibility policy.

Numerical rule: no tolerance change. Existing operation-specific tests remain
authoritative.

Suspected root cause: scientific and source-compatibility implementation
preceded public API productization.

Acceptance:

- commit and check a versioned Rust public inventory;
- organize it by scientific workflow rather than source package history;
- document ownership, precision, errors, coordinates, and stability;
- give every stable workflow a direct test or example;
- require an explicit contract-version update for a stable breaking change.

Related work: the native-language API design PR and the Python, Julia, and CI
issues below.

## Provide a first-class Python-native Thouless API

Labels: `native-api`, `test-gap`

Missing capability: a supported Python-native API. The installed package
currently exports only `thouless._core`.

Affected interface and tests:

- `python/thouless`;
- proposed `python-tests/native`;
- proposed `spec/coverage/python-native.toml`;
- all existing PythTB 2.0 and Kwant 1.5 compatibility tests.

Reproduction:

```console
python -m pip install maturin numpy scipy tinyarray
maturin develop
python -c 'import thouless; print(thouless.__all__)'
```

Expected: `import thouless` exposes documented objects and operations following
the Rust scientific model, without importing `_core`.

Actual: the PyO3 bridge is internal and primarily serves compatibility code.

Numerical rule: use the same operation-specific tolerances and physical
invariants as the Rust implementation.

Suspected root cause: PyO3 was introduced as a source-compatibility transport
boundary before a native Python product surface was designed.

Acceptance:

- implement documented public Python modules over the Rust core;
- keep `_core` internal;
- cover every stable scientific workflow in `python-native.toml`;
- test arrays, complex values, errors, ownership, and GIL behavior;
- build and install a wheel in a clean CI environment;
- provide public type information;
- keep the complete PythTB and Kwant suites green.

Related work: the Rust contract, Julia, and cross-language CI issues.

## Add a stable C ABI and first-class Thouless.jl package

Labels: `native-api`, `test-gap`

Missing capability: a language-neutral ABI and Julia-native package.

Affected interface and tests:

- proposed `crates/thouless-capi`;
- proposed `julia/Thouless`;
- proposed `spec/coverage/julia-native.toml`;
- C ABI ownership, error, array, and panic-containment tests;
- `julia/Thouless/test/runtests.jl`.

Reproduction:

```console
find . -name 'Project.toml' -o -name '*.jl'
rg 'extern "C"|no_mangle|cbindgen' crates src
```

Expected: a versioned C ABI exposes stable Rust workflows, and `Thouless.jl`
wraps them with Julia objects, arrays, exceptions, and finalizers.

Actual: no public ABI, header, Julia source, Julia package metadata, artifact,
or Julia CI exists.

Numerical rule: use Rust operation-specific tolerances and invariant-based
cross-language checks.

Suspected root cause: the initial interoperability target was Python source
compatibility only.

Acceptance:

- define an ABI version and generated header;
- document and test handles, destruction, errors, strings, complex values,
  shapes, strides, allocation, thread safety, and panic containment;
- implement Julia-native objects for every stable workflow;
- use safe Julia finalizers and ordinary Julia array semantics;
- test Julia LTS and the current stable release against the actual native
  artifact on supported platforms.

Related work: the Rust contract, Python, and cross-language CI issues.

## Add cross-language artifact and semantic-parity CI

Labels: `test-gap`

Missing capability: CI for installable language-native artifacts and semantic
parity among Rust, Python, and Julia.

Affected interface and tests:

- `.github/workflows/ci.yml`;
- proposed Python and Julia native suites;
- proposed shared generated conformance cases;
- language-native API and coverage manifests.

Reproduction:

```console
sed -n '1,220p' .github/workflows/ci.yml
```

Expected: required CI builds, clean-installs, and tests all supported language
packages, while a shared suite verifies consistent scientific semantics.

Actual: CI uses Rust on Linux/macOS and Python 3.12/Linux with
`maturin develop`; it has no wheel-install, Julia, C ABI, parity, release
artifact, or Windows binding test.

Numerical rule: use existing operation-specific tolerances. Any new tolerance
requires a scientific justification; source tolerances may not be relaxed.

Suspected root cause: CI grew around implementation and source compatibility
before first-class language APIs were in scope.

Acceptance:

- add required Rust, Python wheel, Julia package, ABI, and parity jobs;
- retain the complete PythTB and Kwant source-test gate;
- cover supported platforms and language versions at the documented PR or
  release cadence;
- install produced artifacts rather than relying on source-tree development
  builds;
- add scheduled ownership, scale, allocation, and performance checks;
- keep evaluator-owned held-out tests outside public CI;
- link every failure or intentional skip to a reproducible issue.

Related work: the Rust contract, Python, and Julia issues.
