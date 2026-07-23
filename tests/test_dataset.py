
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RigidTransform as Tf

from mapfree_refine.data.dataset import MapFreeDataset

SCENE_ROOT = Path("/home/colin/Documents/mapfreechallangedatasets/sample/sample/train/s00000/")  # <-- adjust
GOLDEN = Path(__file__).parent / "golden_s00000.npz"

needs_data = pytest.mark.skipif(
    not (SCENE_ROOT.exists() and GOLDEN.exists()),
    reason="sample scene or golden file not available",
)

@needs_data
def test_loader_matches_golden():
    g = np.load(GOLDEN)
    ds = MapFreeDataset(SCENE_ROOT._str)

    records = {rec["pair_names"][1]: rec for rec in ds}

    for name, K_gold, T_gold in zip(g["names"], g["Ks"], g["T_rels"]):
        bare = str(name).split("/")[-1]
        assert bare in records, f"loader produced no record for {name}"
        rec = records[bare]

        np.testing.assert_allclose(
            rec["image1_k"], K_gold, atol=1e-3,
            err_msg=f"K mismatch for {name}",
        )
        np.testing.assert_allclose(
            rec["T_query_ref"].as_matrix(), T_gold, atol=1e-5,
            err_msg=f"relative pose mismatch for {name} "
                    "(suspects, in order: composition direction, quaternion "
                    "order, world->camera assumption)",
        )


def test_quaternion_roundtrip():
    """poses.txt-style wxyz values must survive parse -> as_quat(scalar_first)."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    if q[0] < 0:          
        q = -q
    t = rng.normal(size=3)

    T = Tf.from_components(t, R.from_quat(q, scalar_first=True))
    q_back = T.rotation.as_quat(scalar_first=True)
    if q_back[0] < 0:
        q_back = -q_back

    np.testing.assert_allclose(q_back, q, atol=1e-10)
    np.testing.assert_allclose(T.translation, t, atol=1e-10)


def test_pose_inverse_is_identity():
    rng = np.random.default_rng(1)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    T = Tf.from_components(rng.normal(size=3), R.from_quat(q, scalar_first=True))
    np.testing.assert_allclose((T * T.inv()).as_matrix(), np.eye(4), atol=1e-10)


def test_intrinsics_project_backproject_roundtrip():
    K = np.array([[551.2, 0, 271.2], [0, 551.2, 350.5], [0, 0, 1.0]])
    rng = np.random.default_rng(2)
    X = rng.uniform([-2, -2, 1.0], [2, 2, 8.0], size=(50, 3))   # points in front

    # project: x = K @ (X / Z), keep pixel coords + depth
    x_h = (K @ (X / X[:, 2:3]).T).T
    px, depth = x_h[:, :2], X[:, 2]

    # backproject: X = depth * K^-1 @ [u, v, 1]
    ones = np.ones((len(px), 1))
    rays = (np.linalg.inv(K) @ np.hstack([px, ones]).T).T
    X_back = rays * depth[:, None]

    np.testing.assert_allclose(X_back, X, atol=1e-9)
