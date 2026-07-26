# Scientific Software Reimplementation Instructions

## 1. Objective

Build Thouless as a complete Rust-native scientific software system for static
tight-binding models, topology, quantum geometry, intrinsic response, finite
systems, and steady-state quantum transport.

The Rust-native API must be designed from scientific objects, invariants, and
real workflows. It must not be a renamed copy of a source package.

Provide source-level compatibility layers for every in-scope public interface
of:

- PythTB 2.0;
- Kwant 1.5.

Both compatibility layers and Rust callers must use the same Rust scientific
core.

The following are not compatibility targets unless the user explicitly changes
the scope:

- Tkwant or explicit time evolution;
- WannierBerri;
- DFTB+ or self-consistent density-functional theory;
- molecular dynamics;
- many-body solvers.

Wannier90 may be supported as an interoperability and model-input boundary.
Other scientific packages may be used as independent references or conformance
sources without becoming compatibility targets.

Treat the work as one complete objective. Implementation order does not create
separate completion targets. If any required capability, source interface,
test, or validation boundary remains open, the project status is `Incomplete`.

## 2. Required Architectural Order

Use this dependency order:

1. Establish the complete scientific and source-interface scope.
2. Design a functionally complete Rust-native API from first principles.
3. Implement reusable Rust core capabilities.
4. Define stable language and data interoperability boundaries.
5. Implement thin PythTB and Kwant compatibility layers.
6. Run all designated source-interface tests through those layers and make a
   reasonable effort to pass each test.
7. Record every unresolved gap in a GitHub issue.
8. Have an independent evaluator run isolated held-out tests on unseen models
   and workflows.

Do not design the native API one source method at a time. One native capability
or API composition may support many source interfaces.

## 3. Establish and Preserve Complete Scope

Before implementing a capability, inspect the applicable:

- public modules, types, functions, parameters, return values, and errors;
- tests, examples, tutorials, and documentation;
- file formats and runtime modes;
- optional numerical backends;
- serial, parallel, accelerated, and scientific-scale behavior;
- real workflows that compose multiple interfaces;
- CI jobs, current GitHub issues, and existing coverage rows.

Record exact source versions or commits, allowed reference materials, source
test locations or manifests, build commands, test commands, comparison rules,
and toolchain requirements.

Maintain these matrices:

- `spec/coverage/native.toml`: scientific capabilities and workflows, reusable
  Rust implementation, native API or API composition, direct validation,
  status, and issue.
- `spec/coverage/pythtb.toml`: every in-scope PythTB interface, corresponding
  Rust capability, designated tests, actual result, status, and issue.
- `spec/coverage/kwant.toml`: every in-scope Kwant interface, corresponding
  Rust capability, designated tests, actual result, status, and issue.

The compatibility matrices must eventually cover every in-scope public source
interface and every corresponding designated source test. A source capability
without a test remains in scope.

Do not:

- remove an unimplemented capability from scope;
- treat missing tests as evidence that a capability is unnecessary;
- relabel unfinished work as a non-goal;
- infer completion from a green subset of tests;
- describe a scaffold or partial implementation as the complete package.

## 4. Design the Rust-native API from First Principles

Derive the native API from:

- physical domain objects, states, and invariants;
- the production, transformation, and consumption of scientific data;
- ownership and mutation semantics;
- numerical precision and error behavior;
- sparse and dense representations;
- serial, parallel, and accelerated execution models;
- workflows spanning periodic, finite, and open-system representations.

The native API must cover the complete scientific capability set as a whole.
It does not need a one-to-one native counterpart for each PythTB or Kwant
interface.

Do not copy historical argument layouts, mutable state conventions, module
boundaries, or implementation details merely to simplify compatibility tests.
Do not distort the native API to make known fixtures easier to pass.

Prefer a minimal complete set of stable scientific abstractions over a large
common framework. Add an abstraction only when a real workflow, invariant, or
backend boundary requires it.

The accepted cross-language target is defined in
[`docs/native-language-api-design.md`](../docs/native-language-api-design.md).
Maintain
[`spec/api/thouless-native-languages.toml`](../spec/api/thouless-native-languages.toml)
as the executable mapping from every Rust scientific capability to its target
Rust, Python, and Julia namespaces. Python must use the PyO3 extension without
making the private `_core` module the user API. Julia must use a versioned C ABI
and must not require Python or C++ as an intermediate runtime.

## 5. Implement One Reusable Scientific Core

Identify and implement shared:

- mathematical objects and coordinate conventions;
- Hamiltonian and operator representations;
- geometry and boundary transformations;
- eigensolvers and linear solvers;
- topology and quantum-geometry algorithms;
- Green-function, mode, and scattering algorithms;
- observables and response calculations;
- input/output and interoperability facilities;
- execution backends.

Higher-level behavior must compose these reusable capabilities. Do not maintain
separate Python scientific algorithms for PythTB and Kwant compatibility.
Direct Rust-native tests must validate the core independently of compatibility
tests.

## 6. Keep Compatibility Layers Thin

Use this dependency structure:

```text
PythTB caller or test -> Thin Python layer --+
                                             |
Kwant caller or test  -> Thin Python layer --+-> Stable boundary -> Rust core
                                             |
Rust caller           -> Rust-native API -----+
```

Compatibility layers may perform only:

- data conversion;
- state mapping;
- error translation;
- Python runtime integration;
- calls across the interoperability boundary.

Define ownership, array layout, complex-number representation, strings,
callbacks, concurrency, lifetimes, and error propagation at the boundary.

Provide source-level compatibility by default. Binary compatibility is required
only when the user explicitly requests it.

Compatibility tests must prove that imports and execution paths exclude the
replaced source implementation. Passing tests against an installed original
package are not migration evidence.

## 7. Use Tests as Evidence, Not as the Specification

For each source baseline, define:

- an exact upstream commit;
- designated test directories or a test manifest;
- build and execution commands;
- runtime and native dependency requirements;
- permitted source-test modifications;
- existing numerical tolerances and comparison rules;
- GitHub Actions job names and artifact locations;
- pass, fail, skip, and error counts.

Keep designated source tests unchanged whenever required. Only packaging,
build, import, or link changes needed to route them through the compatibility
layer are permitted.

Every designated source test must be run. Make a reasonable effort to pass each
test. Full source-test success is not required for an honest progress report,
but every remaining failure or error keeps the project `Incomplete` and must
have a linked GitHub issue.

Never silently delete, rename, weaken, hide, or skip a failing test. An
intentional skip is allowed only when it is visible, justified, and linked to
an open issue.

Source-interface tests do not replace:

- direct Rust-native tests;
- property, invariant, and metamorphic tests;
- differential checks on newly generated inputs;
- end-to-end scientific workflows;
- realistic-scale and backend validation.

Do not invent universal tolerances or performance thresholds. Define and
justify them before the corresponding validation is used for acceptance.

## 8. Use GitHub Issues as the Audit Trail

Use this repository's GitHub issues as the authoritative tracker for:

- failed or errored source-interface tests;
- intentionally skipped tests;
- missing compatibility entries;
- incomplete Rust scientific capabilities;
- numerical, performance, backend, or scale gaps;
- externally blocked validation.

One issue may cover multiple tests only when they share the same general root
cause. Link every affected coverage row to the issue.

Each issue must include:

1. The missing general capability or failure category.
2. Affected interfaces and exact test identifiers.
3. Source version or commit and relevant Thouless commit.
4. A minimal reproduction command.
5. Expected and actual behavior.
6. Existing tolerance or comparison rule for numerical failures.
7. Logs and links to the GitHub Actions run and job.
8. The suspected root cause.
9. Acceptance criteria for the general fix.
10. Related pull requests, commits, and issues.

Use the existing labels:

- `native-api`;
- `compat:pythtb`;
- `compat:kwant`;
- `test-gap`;
- `held-out`.

Keep an issue open until the general fix is merged and the relevant CI evidence
passes. Reopen it or create a regression issue if the failure returns.

For held-out failures, disclose only the permitted failure category and
non-revealing diagnostics. Never place hidden inputs, expected outputs, answer
keys, or revealing traces in this repository.

## 9. Prohibit Fitting to Known Tests

The implementation must not:

- branch on test names, file paths, fixture identities, or execution order;
- return stored results for known inputs, sizes, or parameter combinations;
- embed expected outputs, fixture summaries, hashes, or data fingerprints;
- add example-specific branches without a scientific or public-contract basis;
- weaken tolerances or modify reference results to make tests pass;
- skip tests to hide missing behavior.

Generality evidence must include:

- native tests across sizes, parameters, boundaries, and data types;
- property, invariant, and metamorphic tests;
- differential checks on newly generated inputs;
- at least one end-to-end workflow composed from multiple core capabilities;
- inspection confirming that public test data did not enter implementation
  logic;
- isolated held-out evaluation.

## 10. Keep Held-out Validation Isolated

An independent evaluator must own and run held-out tests that are invisible to
the implementer and inaccessible from the implementation environment.

Held-out tests must:

- use unseen models, inputs, parameters, boundaries, disorder, invariants, and
  cross-interface workflows;
- differ materially from public tests;
- obtain expectations independently of the current Rust implementation;
- reveal only the minimum failure category needed to improve a general
  capability;
- prevent reconstruction of hidden examples from diagnostics.

Public test success proves only the public validation surface. Held-out success
is required before claiming that no obvious public-test fitting remains.

## 11. Report Status Precisely

Use only:

- `Incomplete`: any native capability, compatibility entry, designated test,
  validation requirement, or linked gap issue remains open.
- `Complete`: both coverage matrices are closed; native, compatibility,
  scientific, scale, and held-out validation pass; and all associated gap
  issues are closed.
- `Blocked`: an external permission, missing artifact, or unavailable
  environment prevents further progress, with reproducible evidence and an
  explicit unblock condition.

Report these dimensions separately:

- Rust-native capability coverage;
- source-interface coverage;
- public test pass, fail, skip, and error counts;
- scientific and realistic-scale validation;
- held-out status;
- open GitHub issues.

If held-out validation has not run, report `Not validated` for that dimension
and keep the overall project status `Incomplete`.

## 12. Required Workflow for Every Change

1. Read `README.md`, this instruction, the applicable coverage matrices, and
   linked issues.
2. Identify the general scientific capability and all affected source
   interfaces.
3. Update or add coverage rows before claiming the scope is closed.
4. Implement the smallest complete general mechanism in the Rust core.
5. Keep compatibility work limited to the thin boundary.
6. Add direct native tests and applicable compatibility tests.
7. Run the relevant local checks and GitHub Actions jobs.
8. Create or update issues for every unresolved result.
9. Report actual coverage and test counts without inferring completion.

## 13. Final Check

Before declaring work complete, verify that:

- the native API follows scientific workflows rather than source API history;
- reusable core behavior is implemented once in Rust;
- every affected source interface is represented in its compatibility matrix;
- designated source tests execute the Rust core and exclude the original
  implementation;
- no test was silently weakened, hidden, or skipped;
- unresolved rows and test results link to open issues;
- native tests cover invariants and more than known source fixtures;
- held-out data remain isolated;
- no partial success is described as complete.
