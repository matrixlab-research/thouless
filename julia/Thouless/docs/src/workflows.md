# Workflow modules

The Julia API groups scientific operations by domain. Each wrapper calls the
same Rust implementation used by the Rust and Python interfaces.

- `AD`, `Spectrum`, and `KPM` cover differentiation and spectral workflows.
- `Geometry`, `Continuum`, and `Visualization` cover model geometry and fields.
- `Topology`, `Wannier`, and `Response` cover gauge-covariant band geometry.
- `Observables` and `Transport` cover local operators and open systems.
- `Symmetry`, `Random`, `Graph`, and `LinearAlgebra` provide reusable
  mathematical building blocks.

Every exported binding is listed in the [API reference](reference.md).
