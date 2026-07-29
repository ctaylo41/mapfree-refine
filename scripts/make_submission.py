import zipfile
import argparse
from pathlib import Path
import numpy as np
from mapfree_refine.cache.cache import CorrespondanceCache
from mapfree_refine.solvers.solvers import Solvers
from mapfree_refine.data.dataset import MapFreeDataset


def line_for(frame_id, q_wxyz, t, conf):
    return (f"{frame_id} "
            f"{q_wxyz[0]} {q_wxyz[1]} {q_wxyz[2]} {q_wxyz[3]} "
            f"{t[0]} {t[1]} {t[2]} {conf}")


def main(args):
    cache = CorrespondanceCache(args.cache)
    val_root = Path(args.val)

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene_dir in sorted(val_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            sid = scene_dir.name
            if not cache.exists(sid):
                print(f"no cache for {sid}, skipping")
                continue

            ds = MapFreeDataset(str(scene_dir))
            records = cache.read_scene(sid)
            lines = []

            for r in records:
                rec = ds[r["pair_id"]]
                frame_id = "seq1/" + rec["pair_names"][1]

                if args.gt:
                    T = rec["T_query_ref"]
                    q = T.rotation.as_quat(scalar_first=True)
                    t = T.translation
                    conf = 1.0
                else:
                    out = Solvers.pnp(r["X0"], r["xy1"], r["K1"])
                    if out is None:
                        q = np.array([1.0, 0.0, 0.0, 0.0])
                        t = np.zeros(3)
                        conf = 0.0
                    else:
                        T, inliers = out
                        q = T.rotation.as_quat(scalar_first=True)
                        t = T.translation
                        conf = len(inliers) / len(r["X0"])

                lines.append(line_for(frame_id, q, t, conf))

            zf.writestr(f"pose_{sid}.txt", "\n".join(lines))
            print(f"wrote pose_{sid}.txt ({len(lines)} frames)")

    print(f"done -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="local val correspondence cache dir")
    p.add_argument("--val", required=True, help="local val dataset root (scene dirs)")
    p.add_argument("--out", required=True, help="output zip path")
    p.add_argument("--gt", action="store_true",
                   help="passthrough ground-truth poses for format validation")
    main(p.parse_args())