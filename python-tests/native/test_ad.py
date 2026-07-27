import numpy as np
import pytest

from thouless import ad


def test_affine_projector_gradient_and_jvp_share_the_rust_truth_path():
    base = np.array([[-0.8, 0.25 - 0.1j], [0.25 + 0.1j, 0.9]], dtype=np.complex128)
    directions = np.array(
        [
            [[1.0, 0.0], [0.0, -1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
        ],
        dtype=np.complex128,
    )
    parameters = np.array([0.1, -0.05, 0.08])
    direction = np.array([0.2, 0.4, -0.3])
    target_vector = np.array([1.0, 0.0], dtype=np.complex128)
    target = np.outer(target_vector, target_vector.conj())

    value, gradient = ad.affine_projector_value_and_grad(
        base,
        directions,
        parameters,
        occupied=1,
        target=target,
        minimum_gap=1.0e-5,
    )
    jvp_value, directional = ad.affine_projector_jvp(
        base,
        directions,
        parameters,
        direction,
        occupied=1,
        target=target,
        minimum_gap=1.0e-5,
    )

    assert value == jvp_value
    assert np.dot(gradient, direction) == pytest.approx(directional, rel=2.0e-12)
    step = 1.0e-6
    positive, _ = ad.affine_projector_value_and_grad(
        base,
        directions,
        parameters + step * direction,
        occupied=1,
        target=target,
        minimum_gap=1.0e-5,
    )
    negative, _ = ad.affine_projector_value_and_grad(
        base,
        directions,
        parameters - step * direction,
        occupied=1,
        target=target,
        minimum_gap=1.0e-5,
    )
    assert directional == pytest.approx((positive - negative) / (2.0 * step), rel=2.0e-6)


def test_affine_projector_reports_gap_closure():
    base = np.zeros((2, 2), dtype=np.complex128)
    directions = np.array([[[1.0, 0.0], [0.0, -1.0]]], dtype=np.complex128)
    target = np.diag([1.0, 0.0]).astype(np.complex128)
    with pytest.raises(Exception, match="gap"):
        ad.affine_projector_value_and_grad(
            base,
            directions,
            [0.0],
            occupied=1,
            target=target,
            minimum_gap=1.0e-5,
        )
