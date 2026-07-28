# Rust-native API design for the 100 domain workflows

## Status and evidence boundary

This is an additive API proposal, not an implemented or stable contract.

- Scientific source: the 100-question catalog in
  `matrixlab-research/thouless-benchmark` at commit
  `62f5e9b9ca7b3810275fa088637f0adae74aca7d`.
- Thouless implementation baseline:
  `341a15d6cd855d9e2ab45c6d586f80e2e35bab56`.
- Machine-readable design coverage:
  [`spec/api/thouless-domain-100.toml`](https://github.com/matrixlab-research/thouless/blob/main/spec/api/thouless-domain-100.toml).
- Implementation tracking:
  [issue #17](https://github.com/matrixlab-research/thouless/issues/17).

The benchmark currently records complete public Thouless witnesses for 67 of
the 100 questions. The remaining 33 reduce to eight reusable gaps rather than
33 application-specific functions: targeted sparse Hermitian eigenpairs,
sparse non-Hermitian eigenpairs, real-time propagation, self-consistent
interactions, sparse real-space topology, non-Bloch topology, general physical
parameter differentiation, and constrained inference.

This proposal shows that all 100 questions are expressible through a small
composable API. It does not claim that the corresponding physics, numerical
scale, language bindings, or held-out validation are implemented.

## Decision

Keep the existing scientific modules and 27 stable workflows. Add one narrow
composition boundary with seven public concepts:

1. a parameterized `ModelFamily` that binds typed physical parameters to an
   immutable model;
2. concrete typed system views such as periodic, finite, open, driven, and
   self-consistent systems;
3. a typed scientific `Request<S>` with one output type;
4. lightweight study axes built from ordinary Rust iterators;
5. an optional `SolveContext` for accuracy, resources, backends, and reusable
   state;
6. a `Report<T>` that carries the value, validity, convergence, error budget,
   resources, and provenance; and
7. optional `ParameterSpace` and scalar `Objective` interfaces for
   differentiation and inference.

The ordinary path remains:

```rust,ignore
let bands = thouless::solve(&model, Bands::along(path))?;
let energies = &bands.value.energies;
```

No workflow graph, global registry, string-keyed parameter store, optimizer,
AD tape, or solver cache is required for this path.

## Why these concepts are sufficient

A scientific calculation needs:

- a physical system;
- a question asked of that system;
- optional repeated evaluation over parameters, disorder, size, mesh, or time;
- a controlled numerical execution;
- a result with enough evidence to judge validity; and
- optionally, derivatives or inference over continuous physical controls.

Those are the seven concepts above. The 100 questions add concrete models,
requests, axes, and acceptance conditions, but they do not add new kinds of
software composition.

## 1. Models and parameterized families

Fixed forward calculations continue to accept ordinary immutable models.
Parameterization is optional:

```rust,ignore
pub trait ModelFamily {
    type Parameters: ParameterSpace;
    type Model;

    fn bind(
        &self,
        parameters: &Self::Parameters,
    ) -> Result<Self::Model, ModelError>;
}
```

`ModelFamily` separates discrete structure from continuous physical
coordinates. A family owns lattice connectivity, basis conventions, sparsity,
provenance, and the rule that turns parameters into a bound model. Its
parameter type contains independent physical coordinates rather than ambient
matrix entries.

The existing `TightBindingModel` remains the normal bound periodic model.
Existing builders remain useful for fixed models. Common parameterized
families may be supplied by Thouless, while user-defined Rust structs may
implement `ModelFamily`.

Model metadata should carry a declared unit system, basis ordering, coordinate
conventions, and provenance. Compile-time dimensional arithmetic is not
required. Values use the model's declared units and every report preserves
that declaration.

## 2. Typed system views

Do not introduce a universal `System` enum containing every physical regime.
Use concrete types and transformations so invalid combinations are difficult
to express:

```text
TightBindingModel                  periodic or translation-free bound model
FiniteSystem<M>                   finite selection with source-site provenance
OpenSystem<M>                     finite device plus typed leads and contacts
DrivenSystem<M, D>                model plus a declared time-dependent drive
MeanFieldProblem<M, I>            model plus interaction and double counting
```

Hermitian, generalized, BdG, and non-Hermitian semantics should be explicit in
the model or operator type and checked by the requests that require them.
Sparse and matrix-free capability is a property of the system's operator
boundary, not a promise inferred from matrix size.

Existing geometry transformations remain concrete operations. New types
should be added only for real physical boundaries:

```rust,ignore
let ribbon = model.finite(selection)?;
let device = OpenSystem::builder(ribbon)
    .lead(left)
    .lead(right)
    .build()?;
let driven = DrivenSystem::new(model, drive)?;
```

Do not replace these operations with a global modifier enum.

## 3. Scientific requests

The common execution boundary is a typed request:

```rust,ignore
pub trait Request<S> {
    type Output;

    fn solve(
        self,
        system: &S,
        context: &mut SolveContext,
    ) -> Result<Report<Self::Output>, Error>;
}

pub fn solve<S, R>(
    system: &S,
    request: R,
) -> Result<Report<R::Output>, Error>
where
    R: Request<S>;

pub fn solve_with<S, R>(
    system: &S,
    request: R,
    context: &mut SolveContext,
) -> Result<Report<R::Output>, Error>
where
    R: Request<S>;
```

`solve` creates a default context chosen from the request and system.
`solve_with` exposes advanced control. A request is a scientific output
contract, not a solver algorithm.

The request families needed by the 100 questions are:

| Module | Representative request types |
| --- | --- |
| `spectrum` | `Bands`, `TargetedStates`, `DensityOfStates`, `FermiSurface` |
| `topology` | `BulkInvariant`, `WilsonFlow`, `LocalMarker`, `NonBlochInvariant` |
| `localization` | `Participation`, `LevelStatistics`, `LocalizationLength`, `MobilityEdge` |
| `observables` | `LocalDensity`, `LocalCurrent`, `Torque`, `Correlation` |
| `response` | `LinearResponse`, `NonlinearResponse`, `OpticalResponse`, `ThermoelectricResponse` |
| `transport` | `Scattering`, `LocalTransport`, `Noise`, `Embedding` |
| `dynamics` | `Evolution`, `FloquetSpectrum`, `Pumping` |
| `mean_field` | `SelfConsistentStates`, `CompetingOrders` |
| `inference` | `Fit`, `Identifiability`, `Design` |

These are output families, not one request per benchmark. For example,
`TargetedStates` serves giant moiré supercells, aperiodic approximants,
localization, and sparse numerical-scale questions. `Evolution` serves
Floquet, Josephson, optical, and scale questions.

Algorithm choices belong in request options or the context:

```rust,ignore
let states = thouless::solve_with(
    &system,
    TargetedStates::around(fermi_energy, 32),
    &mut context,
)?;
```

`TargetedStates` specifies the scientific target. The context may choose dense
diagonalization, Lanczos, LOBPCG, Arnoldi, or shift-invert subject to the
system semantics and resource policy.

## 4. Studies use ordinary Rust composition

Parameter sweeps, disorder ensembles, size scaling, and convergence ladders
are repeated evaluations, not new physics solvers. Keep their API small:

```rust,ignore
pub fn study<I, F, T>(
    axis: I,
    evaluate: F,
) -> Result<StudyReport<T>, Error>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Result<Report<T>, Error>;
```

Provide metadata-preserving iterator helpers:

- `Sweep<P>` for ordered physical parameters;
- `Ensemble<P>` for seeded samples and distribution metadata; and
- `Refinement<P>` for nested mesh, size, broadening, polynomial-order, or
  time-step ladders.

They should remain compatible with ordinary iterator adaptors. Do not create a
workflow DSL.

```rust,ignore
let phase_diagram = study(masses, |mass| {
    let model = family.bind(&Parameters { mass })?;
    solve(&model, BulkInvariant::chern(Occupied::below(0.0)))
})?;
```

An ensemble records seeds and distribution parameters, but seeds and sample
counts remain discrete. A refinement study compares reports and constructs a
separated error budget; it does not silently change tolerances until a desired
answer appears.

## 5. Execution context

The default context should make common calculations easy. Advanced controls
are explicit:

```rust,ignore
let mut context = SolveContext::builder()
    .accuracy(Accuracy::relative(1.0e-10))
    .resources(Resources::sparse_only().memory_limit_gib(16))
    .deterministic(true)
    .build()?;
```

`SolveContext` owns or borrows:

- accuracy and convergence policy;
- dense, sparse, matrix-free, CPU, and future accelerator policy;
- deterministic initialization and parallelism policy;
- memory and wall-time budgets;
- reusable symbolic analysis, factorization, preconditioner, Krylov, or
  checkpoint state; and
- cancellation and progress hooks where a real workflow requires them.

Caches and solver bookkeeping are context internals. They are inspectable
through reports but are not ordinary user arguments.

An automatic algorithm choice must be recorded. A sparse-only policy must fail
with a typed error rather than silently materialize a dense matrix. A solver
switch that invalidates a derivative must be reported.

## 6. Reports are scientific evidence

Every request returns:

```rust,ignore
pub struct Report<T> {
    pub value: T,
    pub validity: Validity,
    pub convergence: Convergence,
    pub error_budget: ErrorBudget,
    pub resources: ResourceUsage,
    pub provenance: Provenance,
}
```

Simple users read `value`. Research workflows can inspect:

- residuals, conservation laws, gauge or symmetry checks;
- convergence histories and independent formulation agreement;
- separated finite-size, integration, truncation, stochastic, and solver
  errors;
- selected algorithm, sparse/dense route, iteration count, allocations, and
  memory high-water mark;
- units, basis, model digest, random seeds, and source provenance; and
- validity boundaries such as a separating gap, causal branch, open channel,
  fixed-point branch, or nonsmooth event.

Domain-specific result types may add structured diagnostics, but they should
reuse these common categories. A scalar value without the evidence required by
its request is not a successful report.

## 7. Differentiation and inference are optional layers

Forward users do not need a parameter space. AD and inference reuse a
`ModelFamily`:

```rust,ignore
pub trait ParameterSpace {
    type Direction;
    type Gradient;

    fn validate(&self) -> Result<(), ParameterError>;
}

pub trait Objective<S>: Request<S, Output = f64> {}
```

The user-facing path is:

```rust,ignore
let objective = Scattering::between(left, right)
    .at(fermi_energy)
    .transmission_objective();

let derivative = thouless::ad::value_and_grad(
    &family,
    &parameters,
    objective,
    &mut context,
)?;
```

`value_and_grad` uses the Rust-native JVP/VJP truth path and returns a
`DerivativeReport`. Primitive ChainRules-style rules, pullbacks, adjoint
solves, checkpoints, and cotangent accumulation remain behind this boundary.
Dense Jacobians are optional; JVPs, VJPs, and Hessian-vector products are
explicit operations with typed validity.

Inference composes physical requests and objectives:

```rust,ignore
let problem = InferenceProblem::new(family, initial_parameters)
    .observe(spectral_data, spectral_objective)
    .observe(transport_data, transport_objective)
    .constraints(constraints);

let fit = solve(&problem, Fit::default())?;
```

The optimizer is replaceable. Physics, derivatives, constraints,
identifiability, uncertainty, and independent forward validation remain
separately testable. Discrete invariants are forward validation targets rather
than differentiable losses.

## Domain examples

### Zero-Chern nonlinear bulk-boundary workflow

One model is reused without inventing a workflow-specific API:

```rust,ignore
let bulk = solve(
    &model,
    NonlinearResponse::hall()
        .chemical_potential(mu)
        .temperature(temperature),
)?;

let ribbon = model.finite(termination)?;
let boundary = solve(&ribbon, LocalDensity::spectral(energy_window))?;

let device = OpenSystem::builder(ribbon)
    .lead(left)
    .lead(right)
    .build()?;
let scattering = solve(&device, Scattering::at(mu))?;
```

The scientific workflow is ordinary Rust composition of response, geometry,
local observable, and transport requests.

### Floquet comparison

```rust,ignore
let driven = DrivenSystem::new(model, drive)?;
let direct = solve(&driven, Evolution::one_period(period))?;
let floquet = solve(&driven, FloquetSpectrum::sambe(harmonics))?;
let comparison = compare(direct, floquet)?;
```

Direct propagation and Sambe calculations are independent request
formulations. Time-origin and harmonic-cutoff studies use `Sweep` and
`Refinement`.

### Competing self-consistent orders

```rust,ignore
let problem = MeanFieldProblem::new(model, interaction)
    .double_counting(double_counting)?;
let branches = solve(
    &problem,
    SelfConsistentStates::from_seeds(initial_orders),
)?;
let comparison = solve(&branches, CompetingOrders::by_free_energy())?;
```

The physical fixed-point contract is stable. Mixing, DIIS, and other
algorithms are context policies.

## Mapping the 20 suites to the API

| TBQ range | Primary composition |
| --- | --- |
| 001–005 | model family + spectrum + studies + reports |
| 006–010 | spectrum + studies + reports |
| 011–015 | field-aware models + spectrum + topology + studies |
| 016–020 | topology + spectrum + studies |
| 021–025 | finite views + spectrum + topology + transport |
| 026–030 | topology + response + transport + optional AD |
| 031–035 | ensembles + spectrum + localization + transport |
| 036–040 | open systems + transport + local observables |
| 041–045 | BdG model semantics + spectrum + transport + dynamics |
| 046–050 | non-Hermitian spectrum + topology + studies |
| 051–055 | driven systems + dynamics + studies |
| 056–060 | mean-field problems + fixed-point reports + studies |
| 061–065 | geometry families + targeted spectrum + observables |
| 066–070 | local observables + response + transport + studies |
| 071–075 | response + dynamics + refinement |
| 076–080 | arbitrary geometry + spectrum + localization + topology |
| 081–085 | finite views + spectrum + observables + transport |
| 086–090 | model families + studies + reports across representations |
| 091–095 | context resource policy + reports across solver families |
| 096–100 | parameter spaces + objectives + inference + reports |

The exact five-question rows and referenced request families are checked in
`spec/api/thouless-domain-100.toml`.

## What is deliberately not public

Do not add these to the ordinary stable surface:

- benchmark identifiers or benchmark-specific entry points;
- a universal workflow graph or task registry;
- string-keyed physics parameters;
- an enum containing all system types, observables, or algorithms;
- internal factorization, preconditioner, Krylov, checkpoint, or pullback
  caches;
- a general AD tape;
- dense ambient Jacobians as the default derivative representation;
- one optimizer as part of scientific truth; or
- silent fallback from a declared sparse or matrix-free route.

## Evolution of the stable contract

This proposal does not modify `spec/api/thouless-native.toml`. A proposed
concept enters the stable inventory only when:

1. its reusable Rust implementation exists;
2. direct native tests cover invariants, generated inputs, and invalid cases;
3. required scientific-scale behavior is demonstrated;
4. its affected capability rows are updated;
5. Python and Julia mappings are designed where the capability is part of the
   shared language contract; and
6. unresolved failures remain linked to open issues.

The overall project remains `Incomplete` while any required capability,
language surface, scientific-scale path, or held-out validation is open.
