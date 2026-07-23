import sys
from pathlib import Path

import numpy as np

N_QUERY_FRAMES = 5  


def quat_wxyz_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def parse_keyed_floats(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out[parts[0]] = np.array([float(v) for v in parts[1:]])
    return out


def pose_to_T(vals):
    """(qw qx qy qz tx ty tz) -> 4x4 world->camera matrix."""
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(*vals[:4])
    T[:3, 3] = vals[4:7]
    return T


def K_from_line(vals):
    fx, fy, cx, cy = vals[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def main():
    scene = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("golden.npz")

    intrinsics = parse_keyed_floats(scene / "intrinsics.txt")
    poses = parse_keyed_floats(scene / "poses.txt")

    ref_key = sorted(k for k in poses if k.startswith("seq0/"))[0]
    query_keys = sorted(k for k in poses if k.startswith("seq1/"))[:N_QUERY_FRAMES]

    T_ref = pose_to_T(poses[ref_key])
    T_ref_inv = np.linalg.inv(T_ref)

    names, Ks, T_rels = [], [], []
    for qk in query_keys:
        T_q = pose_to_T(poses[qk])
        names.append(qk)                     
        Ks.append(K_from_line(intrinsics[qk]))
        T_rels.append(T_q @ T_ref_inv)

    np.savez(
        out_path,
        ref_name=ref_key,
        names=np.array(names),
        Ks=np.stack(Ks),
        T_rels=np.stack(T_rels),
    )
    print(f"Wrote {out_path} with reference {ref_key} and {len(names)} query frames:")
    for n, T in zip(names, T_rels):
        print(f"  {n}  t = {np.round(T[:3, 3], 4)}")


if __name__ == "__main__":
    main()
