# Model construction

```julia
using Thouless

lattice = Lattice(reshape([1.0], 1, 1), [1])
builder = ModelBuilder(lattice)
site = add_orbital!(builder, "s", [0.0])
set_onsite!(builder, site, 0.0)
add_hopping!(builder, site, site, [1], -1.0)
model = build(builder)

result = eigensystem(model, [0.25])
```

`ModelBuilder` is mutable and single-use. `build` transfers its native state
into an immutable `Model`; subsequent use of the builder raises an error.
Orbital and lattice-axis indices are one-based at the Julia boundary.

Detailed signatures and failure contracts are collected in the
[API reference](reference.md).
