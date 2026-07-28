Thouless Python API
===================

The Python package is a typed, NumPy-oriented interface to the Rust-owned
scientific implementation. Model objects own native Rust state; numerical
workflows return ordinary NumPy arrays and immutable result records.

.. code-block:: python

   import numpy as np
   import thouless

   lattice = thouless.model.Lattice([[1.0]], [0])
   builder = thouless.model.ModelBuilder(lattice)
   site = builder.add_orbital("s", [0.0])
   builder.set_onsite(site, 0.0)
   builder.add_hopping(site, site, [1], -1.0)
   model = builder.build()

   energies = model.eigensystem([0.25]).energies
   assert np.isfinite(energies).all()

API modules
-----------

.. autosummary::
   :toctree: generated
   :recursive:

   thouless.ad
   thouless.continuum
   thouless.geometry
   thouless.graph
   thouless.kpm
   thouless.linalg
   thouless.model
   thouless.observables
   thouless.random
   thouless.response
   thouless.spectrum
   thouless.symmetry
   thouless.topology
   thouless.transport
   thouless.visualization
   thouless.wannier

Errors
------

.. automodule:: thouless.errors
   :members:
