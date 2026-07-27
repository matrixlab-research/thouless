# Python API

The `thouless` wheel exposes the Rust scientific model directly. `_core` is an
implementation module and is not part of the supported user surface.

```python
import numpy as np
import thouless

lattice = thouless.Lattice([[1.0]], [0])
builder = thouless.ModelBuilder(lattice)
orbital = builder.add_orbital("s", [0.0])
builder.set_onsite(orbital, 0.25)
builder.add_hopping(orbital, orbital, [1], -1.0)
model = builder.build()

energies = model.eigensystem([0.0]).eigenvalues
bands = model.band_structure(np.linspace(0.0, 0.5, 101)[:, None])
```

`ModelBuilder` exclusively owns a mutable Rust builder. `build()` consumes that
builder and returns an immutable Rust-owned `Model`. NumPy inputs are converted
explicitly to binary64 real or complex values. Returned arrays own their
storage and use ordinary NumPy layouts.

The public modules follow scientific workflows:

- `ad` exposes Rust-native JVP and VJP workflows without Python-side finite
  differences;
- `model`, `geometry`, and `spectrum` construct periodic or finite systems;
- `kpm` provides kernel-polynomial spectral workflows;
- `topology`, `wannier`, and `response` provide gauge-covariant geometry;
- `observables` and `visualization` evaluate and interpolate local quantities;
- `transport` provides leads, modes, self-energies, and scattering;
- `continuum`, `symmetry`, `random`, `graph`, and `linalg` provide supporting
  mathematical workflows.

Long model eigensystem, band, transformation, topology-response, and integrated
response calls release the Python GIL after input conversion. A call that
requires a Python callback keeps the GIL; the 0.1 native API has no callback
entry point.

Public exceptions derive from `ThoulessError`. Invalid scientific inputs and
array shapes remain distinguishable from numerical solve failures. The Python
API follows the numerical and coordinate rules in `docs/api-stability.md`.
Native derivative semantics and current coverage are documented in
`docs/native-ad.md`.
