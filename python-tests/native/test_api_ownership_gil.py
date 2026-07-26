import sys
import threading
import time

import numpy as np

import thouless


def chain_model() -> thouless.Model:
    builder = thouless.ModelBuilder(thouless.Lattice([[1.0]], [0]))
    orbital = builder.add_orbital("s", [0.0])
    builder.add_hopping(orbital, orbital, [1], -1.0)
    return builder.build()


def test_public_package_hides_private_extension_and_ships_inline_types():
    assert "_core" not in thouless.__all__
    assert {"Lattice", "ModelBuilder", "Model"}.issubset(thouless.__all__)


def test_model_owns_copied_input_after_builder_mutation():
    primitive = np.array([[1.0]])
    builder = thouless.ModelBuilder(thouless.Lattice(primitive, [0]))
    orbital = builder.add_orbital("s", [0.0])
    builder.add_hopping(orbital, orbital, [1], -1.0)
    model = builder.build()
    primitive[0, 0] = 50.0
    np.testing.assert_allclose(model.hamiltonian([0.0]), [[-2.0]], atol=1.0e-12)


def test_long_data_only_band_call_releases_the_gil():
    model = chain_model()
    running = threading.Event()
    ready = threading.Event()
    count = [0]

    def worker() -> None:
        ready.set()
        while running.is_set():
            count[0] += 1

    running.set()
    thread = threading.Thread(target=worker)
    thread.start()
    ready.wait()
    time.sleep(0.01)
    count[0] = 0
    previous = sys.getswitchinterval()
    try:
        sys.setswitchinterval(10.0)
        model.band_structure(np.linspace(0.0, 0.5, 20_000)[:, None])
        running.clear()
    finally:
        sys.setswitchinterval(previous)
        running.clear()
        thread.join()
    assert count[0] > 0
