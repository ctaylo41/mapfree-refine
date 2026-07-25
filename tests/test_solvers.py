import pytest
import numpy as np
from mapfree_refine.solvers.solvers import Solvers


def generate_random_rigid_transform():
    q, _ = np.linalg.qr(np.random.randn(3,3))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = np.random.uniform(-10, 10, size = (3,))
    return q, t

def test_noisy_recovery_with_weights():
    np.random.seed(42)
    N = 200
    R_true, t_true = generate_random_rigid_transform()

    A = np.random.uniform(-5, 5, size=(N,3))
    weights = np.random.uniform(0.1, 1.0, size=(N,))

    noise = np.random.normal(0, 0.01, size=(N, 3))

    B = A @ R_true.T + t_true + noise
    R_rec = Solvers.weighted_procrustes(A,B,weights)
    assert np.allclose(R_rec.translation, t_true, atol=1e-2)
    assert np.allclose(R_rec.rotation.as_matrix(), R_true, atol=1e-2)

    Rot = R_rec.rotation.as_matrix()
    assert np.allclose(Rot @ Rot.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(Rot), 1.0, atol=1e-6)

