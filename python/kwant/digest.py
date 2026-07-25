"""Kwant-compatible deterministic random-access variates."""

from __future__ import annotations

import struct
import subprocess
import tempfile

import numpy as np

from thouless import _core


def str_to_bytes(value):
    """Encode strings as UTF-8 and leave buffer-compatible values unchanged."""
    if isinstance(value, str):
        return value.encode("utf8")
    return value


def _bytes(value):
    return memoryview(str_to_bytes(value)).tobytes()


def uniform2(value, salt=""):
    """Return two deterministic independent values in ``[0, 1)``."""
    return _core.digest_uniform_pair(_bytes(value), _bytes(salt))


def uniform(value, salt=""):
    """Return a deterministic value uniformly distributed in ``[0, 1)``."""
    return uniform2(value, salt)[0]


def gauss(value, salt=""):
    """Return a deterministic standard-normal variate."""
    return _core.digest_gaussian(_bytes(value), _bytes(salt))


def test(n=20000):
    """Stream an ``n`` by ``n`` deterministic sample to ``dieharder``."""
    count = int(n)
    if count < 0 or count != n:
        raise ValueError("n must be a nonnegative integer")
    with tempfile.NamedTemporaryFile() as output:
        for first in range(count):
            for second in range(count):
                sample = struct.pack(
                    "I",
                    int(
                        2**32
                        * uniform(
                            np.asarray(
                                [first, second],
                                dtype=np.int64,
                            )
                        )
                    ),
                )
                output.write(sample)
        output.flush()
        subprocess.call(["dieharder", "-a", "-g", "201", "-f", output.name])


__all__ = ["gauss", "test", "uniform", "uniform2"]
